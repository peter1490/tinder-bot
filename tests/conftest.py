"""Test fixtures: temp config/storage and tiny synthetic ONNX models standing in for the real ones.

The fake detector always reports one face at a fixed spot; the fake embedders map mean colour to a
vector. That is enough to exercise the full pipeline (decode -> align -> embed -> features -> verdict)
offline and deterministically.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tinderbot.config import Config
from tinderbot.storage import Storage

CHROME = os.environ.get("TINDERBOT_TEST_CHROME", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")


def _linear_embedder(path: Path, size: int, out_name: str, dim: int = 512, seed: int = 0) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(seed)
    w = numpy_helper.from_array(rng.normal(size=(3, dim)).astype(np.float32), "W")
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, size, size])
    out = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, ["N", dim])
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "W"], [out_name]),
    ]
    g = helper.make_graph(nodes, "emb", [inp], [out], initializer=[w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.save(m, str(path))


def _fake_scrfd(path: Path, size: int = 640) -> None:
    """Constant-output SCRFD: one confident face at anchor (stride 8) near (200,200) with 5 landmarks."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inits, outs, nodes = [], [], []
    inp = helper.make_tensor_value_info("input.1", TensorProto.FLOAT, [1, 3, size, size])
    for stride in (8, 16, 32):
        n = (size // stride) ** 2 * 2
        sc = np.zeros((n, 1), np.float32)
        bb = np.zeros((n, 4), np.float32)
        kp = np.zeros((n, 10), np.float32)
        if stride == 8:
            cx, cy = 25, 25  # anchor grid cell -> centre (200, 200)
            idx = (cy * (size // stride) + cx) * 2
            sc[idx] = 0.95
            bb[idx] = [10, 12, 10, 12]  # distances /stride -> box 120x160 around (200,200)
            # landmarks offsets (/stride): eyes, nose, mouth
            kp[idx] = [-4, -4, 4, -4, 0, 1, -3, 6, 3, 6]
        for name, arr in ((f"score_{stride}", sc), (f"bbox_{stride}", bb), (f"kps_{stride}", kp)):
            inits.append(numpy_helper.from_array(arr, name + "_c"))
            nodes.append(helper.make_node("Identity", [name + "_c"], [name]))
            outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(arr.shape)))
    # keep the input "used" so runtimes don't prune it
    nodes.append(helper.make_node("Shape", ["input.1"], ["shape_unused"]))
    outs.append(helper.make_tensor_value_info("shape_unused", TensorProto.INT64, [4]))
    g = helper.make_graph(nodes, "scrfd_fake", [inp], outs, initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.save(m, str(path))


@pytest.fixture(scope="session")
def fake_model_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("models")
    det = root / "buffalo_l__det_10g"
    rec = root / "buffalo_l__w600k_r50"
    clip = root / "clip-vit-base-patch32"
    for d in (det, rec, clip):
        d.mkdir()
    _fake_scrfd(det / "det_10g.onnx")
    _linear_embedder(rec / "w600k_r50.onnx", 112, "embedding", seed=1)
    _linear_embedder(clip / "vision_model.onnx", 224, "image_embeds", seed=2)
    return root


@pytest.fixture
def cfg(tmp_path, fake_model_root) -> Config:
    c = Config(data_dir=str(tmp_path / "data"))
    c.ensure_dirs()
    # point the models dir at the synthetic models
    import shutil

    for d in fake_model_root.iterdir():
        shutil.copytree(d, c.models_path / d.name, dirs_exist_ok=True)
    return c


@pytest.fixture
def storage(cfg) -> Storage:
    st = Storage(cfg.db_path)
    yield st
    st.close()


@pytest.fixture
def models(cfg):
    from tinderbot.models.loader import Models

    return Models(cfg)


def make_image(color: tuple[int, int, int], size: tuple[int, int] = (480, 640), noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size[0], size[1], 3), np.uint8)
    img[:] = color
    if noise:
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def encode_jpeg(img: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()
