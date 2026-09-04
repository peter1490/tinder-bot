"""Face detection (SCRFD) + recognition (ArcFace) on onnxruntime, no insightface dependency.

Re-implements the small pre/post-processing of InsightFace's ``scrfd.py`` / ``arcface_onnx.py`` so
the standard ``buffalo_l`` ONNX files run with only numpy + opencv.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .image_utils import l2norm

# ArcFace 112x112 alignment template (5 points: eyes, nose, mouth corners)
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


@dataclass
class Face:
    bbox: np.ndarray          # x1, y1, x2, y2 in original image pixels
    score: float
    kps: np.ndarray           # 5x2 landmarks
    embedding: np.ndarray | None = None

    @property
    def area(self) -> float:
        return float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))


def make_session(path: Path, threads: int = 0):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(str(path), so, providers=providers)


def nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


def decode_scrfd(outputs: list[np.ndarray], height: int, width: int, det_thresh: float,
                 strides: tuple[int, ...] = (8, 16, 32), num_anchors: int = 2):
    """Turn raw SCRFD outputs into (scores, boxes[x1,y1,x2,y2], kps[n,5,2]) in input-image pixels.

    Outputs are grouped by their last dimension (1 = score, 4 = bbox distances, 10 = landmarks) and
    sorted by number of rows so the function is robust to output naming/order across exports.
    """
    scores = sorted([o for o in outputs if o.shape[-1] == 1], key=lambda o: -o.shape[0])
    bboxes = sorted([o for o in outputs if o.shape[-1] == 4], key=lambda o: -o.shape[0])
    kpss = sorted([o for o in outputs if o.shape[-1] == 10], key=lambda o: -o.shape[0])
    all_scores, all_boxes, all_kps = [], [], []
    for stride, sc, bb, kp in zip(strides, scores, bboxes, kpss, strict=False):
        sc = sc.reshape(-1)
        fh, fw = height // stride, width // stride
        ys, xs = np.mgrid[:fh, :fw]
        centers = np.stack([xs, ys], axis=-1).astype(np.float32).reshape(-1, 2) * stride
        centers = np.repeat(centers, num_anchors, axis=0)
        bb = bb.reshape(-1, 4) * stride
        kp = kp.reshape(-1, 10) * stride
        idx = np.where(sc >= det_thresh)[0]
        if idx.size == 0:
            continue
        c = centers[idx]
        d = bb[idx]
        boxes = np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=-1)
        k = kp[idx].reshape(-1, 5, 2)
        k = np.stack([c[:, None, 0] + k[:, :, 0], c[:, None, 1] + k[:, :, 1]], axis=-1)
        all_scores.append(sc[idx])
        all_boxes.append(boxes)
        all_kps.append(k)
    if not all_scores:
        return np.zeros((0,), np.float32), np.zeros((0, 4), np.float32), np.zeros((0, 5, 2), np.float32)
    return np.concatenate(all_scores), np.concatenate(all_boxes), np.concatenate(all_kps)


class SCRFD:
    """SCRFD detector (strides 8/16/32, 2 anchors, distance-to-bbox + 5 landmarks)."""

    def __init__(self, model_path: Path, input_size: int = 640, det_thresh: float = 0.5,
                 nms_thresh: float = 0.4, threads: int = 0):
        self.sess = make_session(model_path, threads)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.input_size = input_size
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh

    def detect(self, bgr: np.ndarray, max_num: int = 0) -> list[Face]:
        ih, iw = bgr.shape[:2]
        scale = self.input_size / max(ih, iw)
        nh, nw = int(ih * scale), int(iw * scale)
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (self.input_size, self.input_size),
                                     (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(self.output_names, {self.input_name: blob})
        scores, boxes, kps = decode_scrfd(outs, self.input_size, self.input_size, self.det_thresh)
        if scores.size == 0:
            return []
        boxes = boxes / scale
        kps = kps / scale
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep = nms(dets, self.nms_thresh)
        faces = [Face(bbox=dets[i, :4], score=float(dets[i, 4]), kps=kps[i]) for i in keep]
        faces.sort(key=lambda f: -f.area)
        return faces[:max_num] if max_num else faces


def align_face(bgr: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """Similarity-transform the face so the 5 landmarks land on the ArcFace template."""
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if M is None:  # degenerate landmarks: fall back to a crop around the landmarks
        x1, y1 = int(kps[:, 0].min() - 20), int(kps[:, 1].min() - 30)
        x2, y2 = int(kps[:, 0].max() + 20), int(kps[:, 1].max() + 20)
        crop = bgr[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
        return cv2.resize(crop, (size, size))
    return cv2.warpAffine(bgr, M, (size, size), borderValue=0.0)


class ArcFace:
    def __init__(self, model_path: Path, threads: int = 0):
        self.sess = make_session(model_path, threads)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else 112

    def embed_batch(self, aligned: list[np.ndarray]) -> np.ndarray:
        if not aligned:
            return np.zeros((0, 512), dtype=np.float32)
        blob = cv2.dnn.blobFromImages(aligned, 1.0 / 127.5, (self.size, self.size),
                                      (127.5, 127.5, 127.5), swapRB=True)
        out = self.sess.run([self.output_name], {self.input_name: blob})[0]
        return l2norm(out.astype(np.float32))

    def embed(self, aligned: np.ndarray) -> np.ndarray:
        return self.embed_batch([aligned])[0]


class FaceEngine:
    """Detect every face in a photo and attach ArcFace embeddings."""

    def __init__(self, det_path: Path, rec_path: Path, threads: int = 0, det_thresh: float = 0.5,
                 min_face_px: int = 40):
        self.detector = SCRFD(det_path, det_thresh=det_thresh, threads=threads)
        self.recognizer = ArcFace(rec_path, threads=threads)
        self.min_face_px = min_face_px

    def analyse(self, bgr: np.ndarray, max_faces: int = 6) -> list[Face]:
        faces = [
            f for f in self.detector.detect(bgr, max_num=max_faces)
            if (f.bbox[2] - f.bbox[0]) >= self.min_face_px and (f.bbox[3] - f.bbox[1]) >= self.min_face_px
        ]
        if not faces:
            return []
        crops = [align_face(bgr, f.kps, self.recognizer.size) for f in faces]
        for f, e in zip(faces, self.recognizer.embed_batch(crops), strict=True):
            f.embedding = e
        return faces


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of a (n,d) and b (m,d)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0] if a.ndim == 2 else 0, b.shape[0] if b.ndim == 2 else 0), dtype=np.float32)
    return l2norm(a) @ l2norm(b).T
