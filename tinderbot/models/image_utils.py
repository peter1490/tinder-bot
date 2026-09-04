"""Small image helpers shared by the model wrappers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_bgr(path: str | Path | bytes | np.ndarray) -> np.ndarray:
    """Load an image as BGR uint8 (from a path, raw bytes or an existing array)."""
    if isinstance(path, np.ndarray):
        return path
    if isinstance(path, (bytes, bytearray)):
        arr = np.frombuffer(path, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image {path!r}")
    return img


def blur_score(bgr: np.ndarray) -> float:
    """Variance of the Laplacian; < ~60 usually means a blurry photo. Normalised to 0..1 (log scale)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if max(g.shape) > 640:
        s = 640 / max(g.shape)
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    v = float(cv2.Laplacian(g, cv2.CV_64F).var())
    return float(np.clip(np.log1p(v) / np.log1p(1000.0), 0.0, 1.0))


def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)
