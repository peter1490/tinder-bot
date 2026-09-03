import numpy as np

from tinderbot.models.face import ARCFACE_TEMPLATE, align_face, cosine_matrix, decode_scrfd, nms


def _outputs(size=640, hits=()):
    outs = []
    for stride in (8, 16, 32):
        n = (size // stride) ** 2 * 2
        sc, bb, kp = np.zeros((n, 1), np.float32), np.zeros((n, 4), np.float32), np.zeros((n, 10), np.float32)
        for (s, cx, cy, score, dists) in hits:
            if s == stride:
                idx = (cy * (size // stride) + cx) * 2
                sc[idx] = score
                bb[idx] = dists
                kp[idx] = [-1, -1, 1, -1, 0, 0, -1, 1, 1, 1]
        outs += [sc, bb, kp]
    # shuffle output order to prove decode is order-agnostic
    return [outs[i] for i in (3, 0, 6, 4, 1, 7, 5, 2, 8)]


def test_decode_scrfd_positions_and_kps():
    outs = _outputs(hits=[(8, 25, 25, 0.9, (10, 12, 10, 12)), (32, 5, 5, 0.7, (2, 2, 2, 2))])
    scores, boxes, kps = decode_scrfd(outs, 640, 640, det_thresh=0.5)
    assert scores.shape == (2,)
    i = int(np.argmax(scores))
    assert np.allclose(boxes[i], [200 - 80, 200 - 96, 200 + 80, 200 + 96])
    assert kps.shape == (2, 5, 2)
    assert np.allclose(kps[i][0], [200 - 8, 200 - 8])  # left eye offset * stride 8
    j = 1 - i
    assert np.allclose(boxes[j], [160 - 64, 160 - 64, 160 + 64, 160 + 64])
    # threshold filters
    scores2, _, _ = decode_scrfd(outs, 640, 640, det_thresh=0.8)
    assert scores2.shape == (1,)


def test_decode_empty():
    scores, boxes, kps = decode_scrfd(_outputs(), 640, 640, 0.5)
    assert scores.size == 0 and boxes.shape == (0, 4) and kps.shape == (0, 5, 2)


def test_nms_suppresses_overlaps():
    dets = np.array([[0, 0, 100, 100, 0.9], [5, 5, 105, 105, 0.8], [300, 300, 400, 400, 0.7]], np.float32)
    assert nms(dets, 0.4) == [0, 2]


def test_align_face_maps_landmarks_to_template():
    img = np.zeros((300, 300, 3), np.uint8)
    # landmarks = template shifted/scaled: alignment must bring them back onto the template
    kps = ARCFACE_TEMPLATE * 2.0 + np.array([40, 60], np.float32)
    out = align_face(img, kps, 112)
    assert out.shape == (112, 112, 3)


def test_cosine_matrix():
    a = np.array([[1, 0], [0, 1]], np.float32)
    b = np.array([[2, 0]], np.float32)
    s = cosine_matrix(a, b)
    assert np.allclose(s, [[1], [0]])
    assert cosine_matrix(np.zeros((0, 0)), b).shape == (0, 1)
