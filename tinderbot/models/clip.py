"""CLIP / SigLIP-family image and text embeddings from ONNX exports (onnxruntime + tokenizers)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .face import make_session
from .image_utils import l2norm


class ClipEngine:
    def __init__(self, model_dir: Path, preprocess: dict, threads: int = 0):
        self.pp = preprocess
        self.vision = make_session(model_dir / "vision_model.onnx", threads)
        self.text = None
        self.tokenizer = None
        if (model_dir / "text_model.onnx").exists():
            self.text = make_session(model_dir / "text_model.onnx", threads)
        if (model_dir / "tokenizer.json").exists():
            from tokenizers import Tokenizer

            self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._v_in = self.vision.get_inputs()[0].name
        self._v_out = self._pick_output(self.vision, preprocess.get("pooled_output", "image_embeds"))
        if self.text is not None:
            self._t_out = self._pick_output(self.text, preprocess.get("pooled_output", "text_embeds"))
            self._t_inputs = [i.name for i in self.text.get_inputs()]
        self.mean = np.array(preprocess.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(preprocess.get("std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
        self.size = int(preprocess.get("size", 224))
        self.crop = bool(preprocess.get("crop", True))
        self._text_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _pick_output(sess, preferred: str) -> str:
        names = [o.name for o in sess.get_outputs()]
        if preferred in names:
            return preferred
        for cand in ("image_embeds", "text_embeds", "pooler_output"):
            if cand in names:
                return cand
        # fall back to the output with the smallest rank (the pooled one)
        return min(sess.get_outputs(), key=lambda o: len(o.shape)).name

    # ---- images --------------------------------------------------------------
    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if self.crop:  # resize shortest side to size, then centre-crop (CLIP)
            s = self.size / min(h, w)
            rgb = cv2.resize(rgb, (max(self.size, round(w * s)), max(self.size, round(h * s))),
                             interpolation=cv2.INTER_CUBIC)
            h, w = rgb.shape[:2]
            y0, x0 = (h - self.size) // 2, (w - self.size) // 2
            rgb = rgb[y0:y0 + self.size, x0:x0 + self.size]
        else:  # SigLIP: plain resize (squash)
            rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        return (x - self.mean) / self.std

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        batch = np.stack([self._prep(im) for im in images]).astype(np.float32)
        out = self.vision.run([self._v_out], {self._v_in: batch})[0]
        return l2norm(out.astype(np.float32))

    def embed_image(self, bgr: np.ndarray) -> np.ndarray:
        return self.embed_images([bgr])[0]

    # ---- text -------------------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if self.text is None or self.tokenizer is None:
            raise RuntimeError("this model has no text tower / tokenizer")
        todo = [t for t in texts if t not in self._text_cache]
        if todo:
            max_len = int(self.pp.get("max_len", 77))
            pad_id = int(self.pp.get("pad_id", 0))
            ids = np.full((len(todo), max_len), pad_id, dtype=np.int64)
            mask = np.zeros((len(todo), max_len), dtype=np.int64)
            for i, enc in enumerate(self.tokenizer.encode_batch(todo)):
                tok = enc.ids[:max_len]
                ids[i, :len(tok)] = tok
                mask[i, :len(tok)] = 1
            feeds = {"input_ids": ids}
            if "attention_mask" in self._t_inputs:
                feeds["attention_mask"] = mask
            out = self.text.run([self._t_out], feeds)[0]
            for t, v in zip(todo, l2norm(out.astype(np.float32)), strict=True):
                self._text_cache[t] = v
        return np.stack([self._text_cache[t] for t in texts])
