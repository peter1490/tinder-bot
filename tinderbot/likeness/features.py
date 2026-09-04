"""Feature extraction for a profile.

For every photo we compute (locally):
  * SCRFD faces + ArcFace embeddings (identity)          -> "does this person look like people I liked?"
  * CLIP image embedding (style, setting, composition)  -> "do the photos look like photos I liked?"
  * zero-shot CLIP prompt scores                         -> user-configurable taste hints
  * sharpness / face-size / group-photo statistics       -> photo quality

Identity features use the profile's *primary face* (the face that recurs across photos), so a
friend in a group photo does not drive the score. All reference vectors come from the local DB.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config
from ..models.face import Face, cosine_matrix
from ..models.image_utils import blur_score, load_bgr
from ..models.loader import Models
from ..storage import ProfileRecord, Storage

FEATURE_KEYS = [
    "face_sim_liked_max", "face_sim_liked_top3", "face_sim_disliked_max", "face_margin",
    "face_knn_liked_frac", "primary_face_sim_liked", "primary_face_sim_disliked",
    "clip_sim_liked_max", "clip_sim_liked_mean", "clip_sim_disliked_max", "clip_margin", "clip_knn_liked_frac",
    "prompt_score",
    "quality_mean", "quality_max", "face_photo_ratio", "group_photo_ratio", "no_face_ratio",
    "face_size_mean", "identity_consistency",
    "photo_count", "bio_len", "bio_keyword_hits", "verified", "age", "distance_km",
]


@dataclass
class PhotoAnalysis:
    photo_id: str
    url: str
    path: Path | None
    width: int
    height: int
    faces: list[Face]
    clip: np.ndarray | None
    quality: float

    @property
    def largest_face(self) -> Face | None:
        return self.faces[0] if self.faces else None


@dataclass
class ProfileAnalysis:
    profile: ProfileRecord
    photos: list[PhotoAnalysis] = field(default_factory=list)

    def face_embeddings(self) -> np.ndarray:
        embs = [f.embedding for p in self.photos for f in p.faces if f.embedding is not None]
        return np.stack(embs) if embs else np.zeros((0, 0), np.float32)

    def clip_embeddings(self) -> np.ndarray:
        embs = [p.clip for p in self.photos if p.clip is not None]
        return np.stack(embs) if embs else np.zeros((0, 0), np.float32)

    def primary_face(self) -> np.ndarray | None:
        """Embedding of the person who recurs the most across the profile's photos."""
        cands = [p.largest_face.embedding for p in self.photos if p.largest_face is not None]
        cands = [c for c in cands if c is not None]
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        m = np.stack(cands)
        sims = cosine_matrix(m, m)
        np.fill_diagonal(sims, 0)
        return m[int(np.argmax(sims.sum(axis=1)))]


def photo_id_for(url: str, profile_id: str, position: int) -> str:
    h = hashlib.sha1(url.encode()).hexdigest()[:16] if url else f"pos{position}"
    return f"{profile_id}_{h}"


class FeatureExtractor:
    def __init__(self, cfg: Config, models: Models, storage: Storage):
        self.cfg = cfg
        self.models = models
        self.storage = storage
        self.liked_faces = np.zeros((0, 0), np.float32)
        self.disliked_faces = np.zeros((0, 0), np.float32)
        self.liked_clip = np.zeros((0, 0), np.float32)
        self.disliked_clip = np.zeros((0, 0), np.float32)
        self._pos_prompts: np.ndarray | None = None
        self._neg_prompts: np.ndarray | None = None
        self.refresh_references()

    # ---- references ---------------------------------------------------------------
    def refresh_references(self) -> None:
        """Reference sets = enrolled folders + every profile you already labelled (auto or manual)."""
        def stack(*parts: np.ndarray) -> np.ndarray:
            parts = [p for p in parts if p.size]
            return np.concatenate(parts) if parts else np.zeros((0, 0), np.float32)

        st = self.storage
        self.liked_faces = stack(st.references("liked", "face"), st.liked_profile_vectors("face", 1))
        self.disliked_faces = stack(st.references("disliked", "face"), st.liked_profile_vectors("face", 0))
        self.liked_clip = stack(st.references("liked", "clip"), st.liked_profile_vectors("clip", 1))
        self.disliked_clip = stack(st.references("disliked", "clip"), st.liked_profile_vectors("clip", 0))

    def reference_summary(self) -> dict[str, int]:
        return {
            "liked_faces": int(self.liked_faces.shape[0]) if self.liked_faces.size else 0,
            "disliked_faces": int(self.disliked_faces.shape[0]) if self.disliked_faces.size else 0,
            "liked_clip": int(self.liked_clip.shape[0]) if self.liked_clip.size else 0,
            "disliked_clip": int(self.disliked_clip.shape[0]) if self.disliked_clip.size else 0,
        }

    def _prompt_vectors(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        pr = self.cfg.likeness.prompts
        if self._pos_prompts is None and (pr.positive or pr.negative):
            try:
                self._pos_prompts = self.models.clip.embed_texts(pr.positive) if pr.positive else np.zeros((0, 0))
                self._neg_prompts = self.models.clip.embed_texts(pr.negative) if pr.negative else np.zeros((0, 0))
            except RuntimeError:  # model without a text tower
                self._pos_prompts, self._neg_prompts = np.zeros((0, 0)), np.zeros((0, 0))
        return self._pos_prompts, self._neg_prompts

    # ---- per-photo analysis ---------------------------------------------------------
    def analyse_photo(self, image: bytes | np.ndarray | Path, photo_id: str, url: str = "",
                      path: Path | None = None) -> PhotoAnalysis:
        bgr = load_bgr(image)
        faces = self.models.face.analyse(bgr)
        clip_vec = self.models.clip.embed_image(bgr)
        return PhotoAnalysis(photo_id=photo_id, url=url, path=path, width=bgr.shape[1], height=bgr.shape[0],
                             faces=faces, clip=clip_vec, quality=blur_score(bgr))

    def analyse_profile(self, profile: ProfileRecord, images: list[bytes | Path], persist: bool = True) -> ProfileAnalysis:
        analysis = ProfileAnalysis(profile=profile)
        if persist:
            self.storage.upsert_profile(profile)
        for i, img in enumerate(images):
            url = profile.photo_urls[i] if i < len(profile.photo_urls) else ""
            pid = photo_id_for(url, profile.id, i)
            local: Path | None = None
            if persist and isinstance(img, (bytes, bytearray)):
                d = self.cfg.photos_path / profile.id
                d.mkdir(parents=True, exist_ok=True)
                local = d / f"{i:02d}.jpg"
                local.write_bytes(img)
            elif isinstance(img, Path):
                local = img
            try:
                pa = self.analyse_photo(img, pid, url, local)
            except ValueError:
                continue
            analysis.photos.append(pa)
            if persist:
                self.storage.upsert_photo(pid, profile.id, url, str(local) if local else None, pa.width, pa.height,
                                         len(pa.faces), pa.quality, position=i)
                for k, f in enumerate(pa.faces):
                    if f.embedding is not None:
                        self.storage.put_embedding(pid, "face", f.embedding, self.models.face_model_name, idx=k,
                                                   meta={"bbox": [float(v) for v in f.bbox], "score": f.score})
                if pa.clip is not None:
                    self.storage.put_embedding(pid, "clip", pa.clip, self.models.clip_model_name)
        return analysis

    # ---- features -----------------------------------------------------------------
    def features(self, a: ProfileAnalysis) -> dict[str, float]:
        f = {k: 0.0 for k in FEATURE_KEYS}
        p = a.profile
        faces = a.face_embeddings()
        clips = a.clip_embeddings()

        # identity vs. reference faces
        if faces.size and self.liked_faces.size:
            s = cosine_matrix(faces, self.liked_faces)
            f["face_sim_liked_max"] = float(s.max())
            top = np.sort(s.ravel())[::-1][:3]
            f["face_sim_liked_top3"] = float(top.mean())
        if faces.size and self.disliked_faces.size:
            f["face_sim_disliked_max"] = float(cosine_matrix(faces, self.disliked_faces).max())
        f["face_margin"] = f["face_sim_liked_max"] - f["face_sim_disliked_max"]
        f["face_knn_liked_frac"] = self._knn_frac(faces, self.liked_faces, self.disliked_faces, k=7)
        primary = a.primary_face()
        if primary is not None:
            if self.liked_faces.size:
                f["primary_face_sim_liked"] = float(cosine_matrix(primary[None], self.liked_faces).max())
            if self.disliked_faces.size:
                f["primary_face_sim_disliked"] = float(cosine_matrix(primary[None], self.disliked_faces).max())

        # whole-photo style vs. reference photos
        if clips.size and self.liked_clip.size:
            s = cosine_matrix(clips, self.liked_clip)
            f["clip_sim_liked_max"] = float(s.max())
            f["clip_sim_liked_mean"] = float(s.max(axis=1).mean())
        if clips.size and self.disliked_clip.size:
            f["clip_sim_disliked_max"] = float(cosine_matrix(clips, self.disliked_clip).max())
        f["clip_margin"] = f["clip_sim_liked_max"] - f["clip_sim_disliked_max"]
        f["clip_knn_liked_frac"] = self._knn_frac(clips, self.liked_clip, self.disliked_clip, k=7)

        # zero-shot prompts
        pos, neg = self._prompt_vectors()
        if clips.size and pos is not None:
            score = 0.0
            if pos.size:
                score += float(cosine_matrix(clips, pos).mean())
            if neg is not None and neg.size:
                score -= float(cosine_matrix(clips, neg).mean())
            f["prompt_score"] = score

        # photo statistics
        n = len(a.photos)
        if n:
            q = [ph.quality for ph in a.photos]
            f["quality_mean"], f["quality_max"] = float(np.mean(q)), float(np.max(q))
            f["face_photo_ratio"] = sum(1 for ph in a.photos if len(ph.faces) == 1) / n
            f["group_photo_ratio"] = sum(1 for ph in a.photos if len(ph.faces) >= 3) / n
            f["no_face_ratio"] = sum(1 for ph in a.photos if not ph.faces) / n
            sizes = [ph.largest_face.area / (ph.width * ph.height) for ph in a.photos if ph.largest_face is not None]
            f["face_size_mean"] = float(np.mean(sizes)) if sizes else 0.0
            f["identity_consistency"] = self._identity_consistency(a)
        f["photo_count"] = float(n)

        # profile text / meta
        bio = (p.bio or "").strip()
        f["bio_len"] = min(len(bio), 500) / 500.0
        f["bio_keyword_hits"] = float(keyword_hits(bio, self.cfg.likeness.liked_bio_keywords))
        f["verified"] = 1.0 if p.verified else 0.0
        f["age"] = float(p.age or 0)
        f["distance_km"] = float(p.distance_km or 0)
        return f

    @staticmethod
    def _knn_frac(q: np.ndarray, liked: np.ndarray, disliked: np.ndarray, k: int) -> float:
        """Fraction of the k nearest reference vectors (pooled) that are liked. 0.5 when unknown."""
        if not q.size or not (liked.size or disliked.size):
            return 0.5
        pool = np.concatenate([x for x in (liked, disliked) if x.size])
        labels = np.concatenate([np.ones(liked.shape[0]) if liked.size else np.zeros(0),
                                 np.zeros(disliked.shape[0]) if disliked.size else np.zeros(0)])
        sims = cosine_matrix(q, pool).max(axis=0)  # best match per reference across the profile's vectors
        k = max(1, min(k, sims.size // 2))         # never vote with (almost) the whole pool
        idx = np.argsort(sims)[::-1][:k]
        return float(labels[idx].mean())

    @staticmethod
    def _identity_consistency(a: ProfileAnalysis) -> float:
        cands = [ph.largest_face.embedding for ph in a.photos if ph.largest_face is not None]
        cands = [c for c in cands if c is not None]
        if len(cands) < 2:
            return 1.0 if cands else 0.0
        m = np.stack(cands)
        s = cosine_matrix(m, m)
        np.fill_diagonal(s, np.nan)
        return float(np.nanmax(s, axis=1).mean())


def keyword_hits(text: str, keywords: list[str]) -> int:
    if not text or not keywords:
        return 0
    t = text.lower()
    return sum(1 for k in keywords if k and re.search(r"\b" + re.escape(k.lower()) + r"\b", t))
