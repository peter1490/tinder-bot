"""Enrol reference images from local folders into the DB (liked / disliked sets).

Typical layout (the old project's folders work as-is):
    img/accepted/*.jpg              -> liked
    img/denied/*.jpg                -> disliked
    facedir/known_faces/<name>/*.jpg -> liked (every face in the image counts)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from ..config import Config
from ..models.image_utils import load_bgr
from ..models.loader import Models
from ..storage import Storage

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(folder: Path) -> Iterable[Path]:
    if folder.is_file():
        yield folder
        return
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in IMAGE_EXT and p.is_file():
            yield p


def enroll_folder(cfg: Config, models: Models, storage: Storage, folder: Path, label: str,
                  all_faces: bool = False, progress: Callable[[str], None] | None = None) -> dict[str, int]:
    """Embed every image in ``folder`` and store it as a reference of ``label`` ('liked'|'disliked')."""
    assert label in ("liked", "disliked")
    stats = {"images": 0, "faces": 0, "skipped": 0}
    for path in iter_images(folder):
        try:
            bgr = load_bgr(path)
        except ValueError:
            stats["skipped"] += 1
            continue
        stats["images"] += 1
        faces = models.face.analyse(bgr)
        if not all_faces and faces:
            faces = faces[:1]  # largest face only: the reference set is "people", not "crowds"
        for i, f in enumerate(faces):
            if f.embedding is not None:
                storage.put_reference(f"{path}#{i}", label, "face", models.face_model_name, f.embedding)
                stats["faces"] += 1
        clip_vec = models.clip.embed_image(bgr)
        storage.put_reference(str(path), label, "clip", models.clip_model_name, clip_vec)
        if progress:
            progress(f"{label}: {path.name} ({len(faces)} face(s))")
    return stats


def reference_centroids(storage: Storage) -> dict[str, np.ndarray | None]:
    out = {}
    for label in ("liked", "disliked"):
        for kind in ("face", "clip"):
            m = storage.references(label, kind)
            out[f"{label}_{kind}"] = m.mean(axis=0) if m.size else None
    return out
