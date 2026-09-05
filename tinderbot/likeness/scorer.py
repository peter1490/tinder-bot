"""Decision layer: hard filters -> weighted prior -> learned logistic model -> verdict with reasons.

The prior works from day one using your enrolled reference folders. Once enough labelled decisions
exist (auto decisions you confirmed or corrected in ``tinderbot review``, plus imports) a logistic
regression over the same features is trained locally and blended in with a weight that ramps up
with the number of examples.

At every retrain the features of all training examples are recomputed from the stored embeddings
against the *current* reference pools (leaving each profile's own vectors out), so an example scored
months ago against three reference photos is described the same way as one scored today.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..config import Config
from ..storage import ProfileRecord, Storage
from .features import FEATURE_KEYS, keyword_hits

if TYPE_CHECKING:
    from .features import FeatureExtractor


@dataclass
class Verdict:
    like: bool
    score: float                       # final probability of "like"
    prior: float
    learned: float | None
    reasons: list[str] = field(default_factory=list)
    uncertain: bool = False
    filtered: bool = False             # rejected by a hard filter
    crush: bool = False                # score above crush.super_like_threshold -> Super Like candidate
    super_crush: bool = False          # score above crush.message_threshold -> Super Like with a note

    @property
    def action(self) -> str:
        return "like" if self.like else "nope"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class Scorer:
    def __init__(self, cfg: Config, storage: Storage, model_path: Path | None = None,
                 extractor: FeatureExtractor | None = None):
        self.cfg = cfg
        self.storage = storage
        self.extractor = extractor
        self.model_path = model_path or (cfg.models_path / "likeness_lr.pkl")
        self.model = None
        self.n_train = 0
        self.stale = False                 # a saved model was found but does not match FEATURE_KEYS
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
        """Hand-weighted logit.

        With both a liked and a disliked pool the identity/style terms use the *margin* (similarity
        to liked minus similarity to disliked) and the k-NN vote: those stay comparable as the pools
        grow, whereas the raw "max similarity to anything I liked" creeps up with every labelled
        profile and would saturate the prior. With a single pool the absolute similarity is used,
        calibrated so ArcFace ~0.30 and CLIP ~0.60 are neutral.
        """
        w = self.cfg.likeness.weights
        reasons: list[str] = []
        z = w.bias
        # identity: prefer the primary-face similarity, fall back to any-face when unavailable
        liked_sim = f.get("primary_face_sim_liked") or f.get("face_sim_liked_max", 0.0)
        disliked_sim = f.get("primary_face_sim_disliked") or f.get("face_sim_disliked_max", 0.0)
        if liked_sim and disliked_sim:
            margin = liked_sim - disliked_sim   # ArcFace: +-0.15 is a clear lean to one side
            z += w.face_sim_liked * margin / 0.15
            reasons.append(f"face≈liked {liked_sim:.2f}" if margin >= 0 else f"face≈disliked {disliked_sim:.2f}")
        elif liked_sim:
            # ArcFace cosine: >0.5 same person, 0.25-0.45 strong resemblance. Rescale so 0.3 is neutral.
            z += w.face_sim_liked * (liked_sim - 0.3) / 0.15
            reasons.append(f"face≈liked {liked_sim:.2f}")
        elif disliked_sim:
            z += w.face_sim_disliked * (disliked_sim - 0.3) / 0.15
            reasons.append(f"face≈disliked {disliked_sim:.2f}")
        knn = f.get("primary_face_knn_liked_frac", f.get("face_knn_liked_frac", 0.5))
        if knn != 0.5:
            z += w.face_sim_liked * (knn - 0.5)
        liked_clip = f.get("clip_sim_liked_mean", 0.0)
        disliked_clip = f.get("clip_sim_disliked_max", 0.0)
        if liked_clip and disliked_clip:
            z += w.clip_sim_liked * f.get("clip_margin", 0.0) / 0.04
            reasons.append(f"photos≈liked {liked_clip:.2f}")
        elif liked_clip:
            # CLIP image-image cosine: 0.6 neutral for portraits, 0.8+ very similar
            z += w.clip_sim_liked * (liked_clip - 0.6) / 0.1
            reasons.append(f"photos≈liked {liked_clip:.2f}")
        elif disliked_clip:
            z += w.clip_sim_disliked * (disliked_clip - 0.6) / 0.1
        cknn = f.get("clip_knn_liked_frac", 0.5)
        if cknn != 0.5:
            z += w.clip_sim_liked * (cknn - 0.5)
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
    @staticmethod
    def _vec(f: dict[str, float]) -> np.ndarray:
        v = np.array([float(f.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32)
        # distance has a long tail (passport / global mode profiles): compress it before scaling
        v[FEATURE_KEYS.index("distance_km")] = math.log1p(max(0.0, float(v[FEATURE_KEYS.index("distance_km")])))
        return v

    def _load_model(self) -> None:
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as fh:
                    obj = pickle.load(fh)
            except Exception:
                self.model = None
                return
            if list(obj.get("keys") or []) != FEATURE_KEYS:
                # trained on another feature layout: predicting with it would be wrong (or crash)
                self.model, self.n_train, self.stale = None, int(obj.get("n", 0)), True
                return
            self.model, self.n_train = obj["model"], obj["n"]

    def training_matrix(self) -> tuple[np.ndarray, np.ndarray, int]:
        """(X, y, n_recomputed) from every labelled decision, features recomputed when possible."""
        ex = self.extractor
        if ex is not None:
            ex.refresh_references()
        X, y, recomputed = [], [], 0
        for r in self.storage.training_examples():
            feats = None
            if ex is not None:
                analysis = ex.analysis_from_storage(r["profile_id"])
                if analysis is not None:
                    feats = ex.features(analysis, exclude_profile_id=r["profile_id"])
                    recomputed += 1
            if feats is None:
                try:
                    feats = json.loads(r["features"])
                except Exception:
                    continue
            X.append(self._vec(feats))
            y.append(int(r["label"]))
        if not y:
            return np.zeros((0, len(FEATURE_KEYS)), np.float32), np.zeros((0,), int), 0
        return np.stack(X), np.array(y), recomputed

    def retrain(self) -> int:
        """Fit the logistic model from every labelled decision. Returns #examples used."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, y, _ = self.training_matrix()
        n = int(len(y))
        self._decisions_since_train = 0
        if n < self.cfg.likeness.learning.min_examples or len(set(y.tolist())) < 2:
            self.model, self.n_train, self.stale = None, n, False   # prior only until more labels exist
            return n
        # unweighted on purpose: the output must be an honest probability (the crush thresholds and the
        # uncertain band read it as one); class_weight="balanced" inflates scores near the threshold
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=1000))
        model.fit(X, y)
        self.model, self.n_train, self.stale = model, n, False
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as fh:
            pickle.dump({"model": model, "n": n, "keys": FEATURE_KEYS}, fh)
        return n

    def learned_weight(self) -> float:
        lr = self.cfg.likeness.learning
        if self.model is None or self.n_train < lr.min_examples:
            return 0.0
        return float(np.clip((self.n_train - lr.min_examples) / max(1, lr.blend_full_at - lr.min_examples), 0.15, 1.0))

    def maybe_retrain(self) -> bool:
        """Count a new decision; retrain (and refresh the reference pools) every ``retrain_every``."""
        self._decisions_since_train += 1
        if self._decisions_since_train >= self.cfg.likeness.learning.retrain_every:
            self.retrain()
            return True
        return False

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
        cr = self.cfg.crush
        trusted = learned is not None or not cr.require_learned
        crush = bool(cr.enabled and like and trusted and score >= cr.super_like_threshold)
        super_crush = bool(crush and score >= cr.message_threshold)
        if super_crush:
            reasons.append("super crush")
        elif crush:
            reasons.append("crush")
        return Verdict(like=like, score=score, prior=prior, learned=learned, reasons=reasons, uncertain=uncertain,
                       crush=crush, super_crush=super_crush)
