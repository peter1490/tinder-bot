"""Lazy construction of the model engines from the config (download on first use)."""

from __future__ import annotations

from functools import cached_property

from ..config import Config
from .clip import ClipEngine
from .face import FaceEngine
from .registry import REGISTRY, ensure_model


class Models:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.ensure_dirs()

    @cached_property
    def face(self) -> FaceEngine:
        det_dir = ensure_model(self.cfg.models_path, self.cfg.models.face_detector)
        rec_dir = ensure_model(self.cfg.models_path, self.cfg.models.face_recognizer)
        det = next(det_dir.glob("*.onnx"))
        rec = next(rec_dir.glob("*.onnx"))
        return FaceEngine(det, rec, threads=self.cfg.models.threads)

    @cached_property
    def clip(self) -> ClipEngine:
        name = self.cfg.models.clip
        d = ensure_model(self.cfg.models_path, name)
        return ClipEngine(d, REGISTRY[name].preprocess, threads=self.cfg.models.threads)

    @property
    def face_model_name(self) -> str:
        return self.cfg.models.face_recognizer

    @property
    def clip_model_name(self) -> str:
        return self.cfg.models.clip
