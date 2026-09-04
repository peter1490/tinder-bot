"""Catalog of local ONNX models and a tiny downloader (first run only, then fully offline).

Sources are plain HTTPS file URLs so no hub SDK is needed. Add your own entries to swap in a
different CLIP/SigLIP export or face model; the loaders only depend on the ``files`` and
``preprocess`` fields.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    files: dict[str, list[str]]  # local filename -> candidate URLs (tried in order)
    # preprocessing parameters used by the CLIP-family loader
    preprocess: dict = field(default_factory=dict)
    notes: str = ""
    # files the loader can live without (e.g. the text tower: only zero-shot prompts need it)
    optional: frozenset[str] = frozenset()


HF = "https://huggingface.co/{repo}/resolve/main/{path}"

CLIP_PP = {
    "size": 224,
    "crop": True,
    "mean": [0.48145466, 0.4578275, 0.40821073],
    "std": [0.26862954, 0.26130258, 0.27577711],
    "max_len": 77,
    "pad_id": 49407,
}


TEXT_FILES = frozenset({"text_model.onnx", "tokenizer.json"})


def _clip_files(repo: str, quantized: bool = False) -> dict[str, list[str]]:
    suffix = "_quantized" if quantized else ""
    return {
        "vision_model.onnx": [HF.format(repo=repo, path=f"onnx/vision_model{suffix}.onnx")],
        "text_model.onnx": [HF.format(repo=repo, path=f"onnx/text_model{suffix}.onnx")],
        "tokenizer.json": [HF.format(repo=repo, path="tokenizer.json")],
    }


REGISTRY: dict[str, ModelSpec] = {
    # --- Face: InsightFace "buffalo_l" pack (SCRFD-10G detector + ArcFace ResNet50 @ WebFace600K) ---
    # Mirrored by the Immich project (actively maintained); original zip on the insightface GitHub release.
    "buffalo_l/det_10g": ModelSpec(
        name="buffalo_l/det_10g",
        files={"det_10g.onnx": [
            HF.format(repo="immich-app/buffalo_l", path="detection/model.onnx"),
            "zip:https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip#det_10g.onnx",
        ]},
        notes="SCRFD 10G face detector with 5-point landmarks (640x640 input).",
    ),
    "buffalo_l/w600k_r50": ModelSpec(
        name="buffalo_l/w600k_r50",
        files={"w600k_r50.onnx": [
            HF.format(repo="immich-app/buffalo_l", path="recognition/model.onnx"),
            "zip:https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip#w600k_r50.onnx",
        ]},
        notes="ArcFace R50 embedding (512-d, 112x112 aligned input).",
    ),
    # --- CLIP family (HF 'Xenova' ONNX exports: vision_model.onnx / text_model.onnx / tokenizer.json) ---
    "clip-vit-base-patch32": ModelSpec(
        name="clip-vit-base-patch32",
        files=_clip_files("Xenova/clip-vit-base-patch32"),
        preprocess=CLIP_PP,
        optional=TEXT_FILES,
        notes="OpenAI CLIP ViT-B/32, 512-d joint space. ~350MB fp32.",
    ),
    "clip-vit-base-patch32-q8": ModelSpec(
        name="clip-vit-base-patch32-q8",
        files=_clip_files("Xenova/clip-vit-base-patch32", quantized=True),
        preprocess=CLIP_PP,
        optional=TEXT_FILES,
        notes="INT8 quantized CLIP ViT-B/32 (~90MB, ~3x faster on CPU, tiny accuracy loss).",
    ),
    "clip-vit-large-patch14": ModelSpec(
        name="clip-vit-large-patch14",
        files=_clip_files("Xenova/clip-vit-large-patch14"),
        preprocess=CLIP_PP,
        optional=TEXT_FILES,
        notes="CLIP ViT-L/14, 768-d, noticeably better similarity, ~1.7GB, slow on CPU.",
    ),
    "siglip-base-patch16-224": ModelSpec(
        name="siglip-base-patch16-224",
        files=_clip_files("Xenova/siglip-base-patch16-224"),
        preprocess={"size": 224, "crop": False, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                    "max_len": 64, "pad_id": 1, "pooled_output": "pooler_output"},
        optional=TEXT_FILES,
        notes="SigLIP (sigmoid loss) base model; sharper image-image similarity than CLIP-B/32.",
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stderr.write(f"\r  {dest.name}: {done / 1e6:.1f}/{total / 1e6:.1f} MB")
        sys.stderr.write("\n")
    tmp.replace(dest)


def _download_from_zip(url: str, member: str, dest: Path) -> None:
    zpath = dest.parent / "_download.zip"
    _download(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith("/" + member) or n == member]
        if not names:
            raise FileNotFoundError(f"{member} not in {url}")
        with z.open(names[0]) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
    zpath.unlink(missing_ok=True)


def model_dir(models_root: Path, name: str) -> Path:
    return models_root / name.replace("/", "__")


def ensure_model(models_root: Path, name: str, quiet: bool = False) -> Path:
    """Return the directory containing the model's files, downloading them the first time."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'. Known: {', '.join(sorted(REGISTRY))}")
    spec = REGISTRY[name]
    d = model_dir(models_root, name)
    d.mkdir(parents=True, exist_ok=True)
    for fname, urls in spec.files.items():
        dest = d / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        errors = []
        for url in urls:
            try:
                if not quiet:
                    print(f"Downloading {name}/{fname} ...", file=sys.stderr)
                if url.startswith("zip:"):
                    zurl, member = url[4:].split("#", 1)
                    _download_from_zip(zurl, member, dest)
                else:
                    _download(url, dest)
                break
            except Exception as e:  # try the next mirror
                errors.append(f"{url}: {e}")
        else:
            msg = (f"could not download {fname} for model {name}. Tried:\n  " + "\n  ".join(errors)
                   + f"\nYou can also place the file manually at {dest}")
            if fname in spec.optional:
                print(f"warning: {msg}\n(optional file: zero-shot text prompts will be disabled)", file=sys.stderr)
                continue
            raise RuntimeError(msg)
    return d


def is_available(models_root: Path, name: str) -> bool:
    spec = REGISTRY.get(name)
    if not spec:
        return False
    d = model_dir(models_root, name)
    return all((d / f).exists() for f in spec.files if f not in spec.optional)
