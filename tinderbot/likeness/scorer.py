"""Decision layer: hard filters -> weighted prior -> learned logistic model -> verdict with reasons.

The prior works from day one using your enrolled reference folders. Once enough labelled decisions
exist (auto decisions you confirmed or corrected in ``tinderbot review``, plus imports) a logistic
regression over the same features is trained locally and blended in with a weight that ramps up
with the number of examples.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config
from ..storage import ProfileRecord, Storage
from .features import FEATURE_KEYS, keyword_hits


@dataclass
class Verdict:
    like: bool
    score: float                       # final probability of "like"
    prior: float
    learned: float | None
    reasons: list[str] = field(default_factory=list)
    uncertain: bool = False
    filtered: bool = False             # rejected by a hard filter

    @property
    def action(self) -> str:
        return "like" if self.like else "nope"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class Scorer:
    def __init__(self, cfg: Config, storage: Storage, model_path: Path | None = None):
        self.cfg = cfg
        self.storage = storage
        self.model_path = model_path or (cfg.models_path / "likeness_lr.pkl")
        self.model = None
        self.n_train = 0
        self._decisions_since_train = 0
        self._load_model()

    # ---- hard filters ----------------------------------------------------------------
    def hard_filter(self, p: ProfileRecord, feats: dict[str, float]) -> str | None:
        lk = self.cfg.likeness
        if p.age is not None and not (lk.min_age <= p.age <= lk.max_age):
            return f"age {p.age} outside [{lk.min_age}, {lk.max_age}]"
        if lk.max_distance_km and p.distance_km and p.distance_km > lk.max_distance_km:
            return f"distance {p.distance_km:.0f} km > {lk.max_distance_km:.0f}"
        if lk.require_verified and not p.verified:
            return "not photo-verified"
        if lk.require_face_photo and feats.get("photo_count", 0) > 0 and feats.get("no_face_ratio", 0) >= 1.0:
            return "no photo with a detectable face"
        hits = keyword_hits(p.bio or "", lk.blocked_bio_keywords)
        if hits:
            return "blocked bio keyword"
        return None

    # ---- prior ----------------------------------------------------------------------
    def prior_logit(self, f: dict[str, float]) -> tuple[float, list[str]]:
        w = self.cfg.likeness.weights
        reasons: list[str] = []
        z = w.bias
        # identity: prefer the primary-face similarity, fall back to any-face when unavailable
        liked_sim = f.get("primary_face_sim_liked") or f.get("face_sim_liked_max", 0.0)
        disliked_sim = f.get("primary_face_sim_disliked") or f.get("face_sim_disliked_max", 0.0)
        if liked_sim:
            # ArcFace cosine: >0.5 same person, 0.25-0.45 strong resemblance. Rescale so 0.3 is neutral.
            z += w.face_sim_liked * (liked_sim - 0.3) / 0.15
            reasons.append(f"face≈liked {liked_sim:.2f}")
        if disliked_sim:
            z += w.face_sim_disliked * (disliked_sim - 0.3) / 0.15
            if disliked_sim > liked_sim:
                reasons.append(f"face≈disliked {disliked_sim:.2f}")
        knn = f.get("face_knn_liked_frac", 0.5)
        if knn != 0.5:
            z += 0.5 * w.face_sim_liked * (knn - 0.5) * 2
        if f.get("clip_sim_liked_max"):
            # CLIP image-image cosine: 0.6 neutral for portraits, 0.8+ very similar
            z += w.clip_sim_liked * (f["clip_sim_liked_mean"] - 0.6) / 0.1
            reasons.append(f"photos≈liked {f['clip_sim_liked_mean']:.2f}")
        if f.get("clip_sim_disliked_max"):
            z += w.clip_sim_disliked * (f["clip_sim_disliked_max"] - 0.6) / 0.1
        cknn = f.get("clip_knn_liked_frac", 0.5)
        if cknn != 0.5:
            z += 0.5 * w.clip_sim_liked * (cknn - 0.5) * 2
        if f.get("prompt_score"):
            z += w.text_prompts * f["prompt_score"] / 0.05
            reasons.append(f"prompts {f['prompt_score']:+.3f}")
        quality = 0.5 * f.get("quality_mean", 0) + 0.5 * f.get("face_photo_ratio", 0) - 0.5 * f.get("group_photo_ratio", 0)
        z += w.photo_quality * (quality - 0.4) * 2
        if f.get("bio_keyword_hits"):
            z += w.bio_keywords * min(3.0, f["bio_keyword_hits"])
            reasons.append(f"bio keywords x{int(f['bio_keyword_hits'])}")
        if f.get("identity_consistency", 1.0) < 0.2 and f.get("photo_count", 0) >= 3:
            z -= 1.0
            reasons.append("inconsistent identity across photos")
        return z, reasons

    # ---- learned model ---------------------------------------------------------------
    def _vec(self, f: dict[str, float]) -> np.ndarray:
        return np.array([float(f.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32)

    def _load_model(self) -> None:
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as fh:
                    obj = pickle.load(fh)
                self.model, self.n_train = obj["model"], obj["n"]
            except Exception:
                self.model = None

    def retrain(self) -> int:
        """Fit the logistic model from every labelled decision. Returns #examples used."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        rows = self.storage.training_examples()
        X, y = [], []
        for r in rows:
            try:
                feats = json.loads(r["features"])
            except Exception:
                continue
            X.append(self._vec(feats))
            y.append(int(r["label"]))
        n = len(y)
        if n < self.cfg.likeness.learning.min_examples or len(set(y)) < 2:
            self.model, self.n_train = None, n
            return n
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500, class_weight="balanced"))
        model.fit(np.stack(X), np.array(y))
        self.model, self.n_train = model, n
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as fh:
            pickle.dump({"model": model, "n": n, "keys": FEATURE_KEYS}, fh)
        self._decisions_since_train = 0
        return n

    def learned_weight(self) -> float:
        lr = self.cfg.likeness.learning
        if self.model is None or self.n_train < lr.min_examples:
            return 0.0
        return float(np.clip((self.n_train - lr.min_examples) / max(1, lr.blend_full_at - lr.min_examples), 0.15, 1.0))

    def maybe_retrain(self) -> None:
        self._decisions_since_train += 1
        if self._decisions_since_train >= self.cfg.likeness.learning.retrain_every:
            self.retrain()

    # ---- verdict ---------------------------------------------------------------------
    def decide(self, p: ProfileRecord, feats: dict[str, float]) -> Verdict:
        lk = self.cfg.likeness
        reason = self.hard_filter(p, feats)
        if reason:
            return Verdict(like=False, score=0.0, prior=0.0, learned=None, reasons=[f"filtered: {reason}"], filtered=True)
        z, reasons = self.prior_logit(feats)
        prior = sigmoid(z)
        learned = None
        w = self.learned_weight()
        if w > 0 and self.model is not None:
            learned = float(self.model.predict_proba(self._vec(feats)[None])[0, 1])
            reasons.append(f"learned {learned:.2f} (w={w:.2f}, n={self.n_train})")
        score = (1 - w) * prior + w * (learned if learned is not None else prior)
        like = score >= lk.like_threshold
        uncertain = abs(score - lk.like_threshold) <= lk.uncertain_band
        if uncertain:
            reasons.append("uncertain")
        return Verdict(like=like, score=score, prior=prior, learned=learned, reasons=reasons, uncertain=uncertain)
