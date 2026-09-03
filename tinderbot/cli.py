"""Command line interface: ``tinderbot --help``."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import Config, load_config

app = typer.Typer(add_completion=False, help="Fully local Tinder auto-swiper (on-device likeness scoring).")
console = Console()

_cfg_opt = typer.Option(None, "--config", "-c", help="Path to config.toml (default: ./config.toml or the example).")


def _cfg(path: str | None) -> Config:
    cfg = load_config(path)
    cfg.ensure_dirs()
    return cfg


def _storage(cfg: Config):
    from .storage import Storage

    return Storage(cfg.db_path)


@app.command()
def login(config: str = _cfg_opt):
    """Open the persistent browser profile so you can log in once (session is kept on disk)."""
    from .browser.driver import launch
    from .browser.tinder_page import TinderPage

    cfg = _cfg(config)
    s = launch(cfg, "https://tinder.com/")
    try:
        tp = TinderPage(s.page)
        console.print(f"Browser backend: [bold]{s.backend}[/bold]. Log in in the window (phone/Google/Facebook).")
        if tp.wait_for_login(max_minutes=20):
            console.print("[green]Logged in. The session is stored in data/browser-profile.[/green]")
        else:
            console.print("[red]Timed out waiting for login.[/red]")
    finally:
        s.close()


@app.command("download-models")
def download_models(config: str = _cfg_opt):
    """Fetch the ONNX models once (afterwards everything runs offline)."""
    from .models.registry import ensure_model

    cfg = _cfg(config)
    for name in (cfg.models.face_detector, cfg.models.face_recognizer, cfg.models.clip):
        d = ensure_model(cfg.models_path, name)
        console.print(f"[green]ok[/green] {name} -> {d}")


@app.command()
def enroll(
    liked: list[Path] = typer.Option([], "--liked", help="Folder(s) of photos of people you like."),
    disliked: list[Path] = typer.Option([], "--disliked", help="Folder(s) of photos you would swipe left on."),
    all_faces: bool = typer.Option(False, help="Use every face in an image (default: largest face only)."),
    reset: bool = typer.Option(False, help="Clear existing reference vectors first."),
    config: str = _cfg_opt,
):
    """Build the reference sets from local folders (e.g. img/accepted, img/denied, facedir/known_faces)."""
    from .likeness.references import enroll_folder
    from .models.loader import Models

    cfg = _cfg(config)
    st = _storage(cfg)
    if reset:
        st.clear_references()
    models = Models(cfg)
    if not liked and not disliked:
        defaults = [("liked", Path("img/accepted")), ("liked", Path("facedir/known_faces")), ("disliked", Path("img/denied"))]
        for label, p in defaults:
            if p.exists():
                (liked if label == "liked" else disliked).append(p)
        if not liked and not disliked:
            raise typer.BadParameter("give --liked/--disliked folders (none of the default folders exist)")
    for label, folders in (("liked", liked), ("disliked", disliked)):
        for folder in folders:
            stats = enroll_folder(cfg, models, st, folder, label, all_faces=all_faces,
                                  progress=lambda m: console.print(f"  {m}"))
            console.print(f"[green]{label}[/green] {folder}: {stats}")


@app.command()
def swipe(
    max_swipes: int = typer.Option(None, "--max", help="Cap for this run (still bounded by the daily budget)."),
    shadow: bool = typer.Option(False, help="Do not swipe: score cards while YOU swipe and learn from your choices."),
    loop: bool = typer.Option(False, help="Keep running sessions with breaks until the daily budget is used."),
    seed: int = typer.Option(None, help="Random seed (for reproducible pacing in tests)."),
    config: str = _cfg_opt,
):
    """Run a swiping session (auto) or a shadow session (learn from you)."""
    from .models.loader import Models
    from .runner import Runner

    cfg = _cfg(config)
    st = _storage(cfg)
    runner = Runner(cfg, st, Models(cfg), seed=seed)
    stats = runner.run_shadow(max_swipes) if shadow else runner.run_auto(max_swipes, once=not loop)
    console.print(f"liked={stats.liked} noped={stats.noped} skipped={stats.skipped} captchas={stats.captchas}")


@app.command()
def score(
    images: list[Path] = typer.Argument(..., help="Photos of one profile (or a folder)."),
    bio: str = typer.Option("", help="Optional bio text."),
    age: int = typer.Option(None),
    config: str = _cfg_opt,
):
    """Score local photos as if they were a profile (offline sanity check of your references/weights)."""
    from .likeness.features import FeatureExtractor
    from .likeness.references import iter_images
    from .likeness.scorer import Scorer
    from .models.loader import Models
    from .storage import ProfileRecord

    cfg = _cfg(config)
    st = _storage(cfg)
    models = Models(cfg)
    ex = FeatureExtractor(cfg, models, st)
    paths = [p for img in images for p in iter_images(img)]
    prof = ProfileRecord(id="local_" + str(int(time.time())), name="local", age=age, bio=bio, photo_urls=[str(p) for p in paths])
    analysis = ex.analyse_profile(prof, paths, persist=False)
    feats = ex.features(analysis)
    v = Scorer(cfg, st).decide(prof, feats)
    t = Table(title=f"verdict: {'LIKE' if v.like else 'NOPE'}  p={v.score:.3f} (prior {v.prior:.3f})")
    t.add_column("feature")
    t.add_column("value", justify="right")
    for k, val in feats.items():
        t.add_row(k, f"{val:.3f}")
    console.print(t)
    console.print("reasons: " + ", ".join(v.reasons))


@app.command()
def retrain(config: str = _cfg_opt):
    """Refit the learned likeness model from all labelled decisions."""
    from .likeness.scorer import Scorer

    cfg = _cfg(config)
    st = _storage(cfg)
    n = Scorer(cfg, st).retrain()
    console.print(f"examples: {n} (learned model {'active' if n >= cfg.likeness.learning.min_examples else 'needs more labels'})")


@app.command()
def review(limit: int = typer.Option(30), config: str = _cfg_opt):
    """Confirm or correct recent uncertain auto-decisions (labels feed the learned model)."""
    cfg = _cfg(config)
    st = _storage(cfg)
    rows = st.uncertain_for_review(limit)
    if not rows:
        console.print("nothing to review")
        return
    for r in rows:
        console.print(f"\n[bold]{r['name']} {r['age'] or ''}[/bold]  auto={r['action']} p={r['score']:.2f}")
        console.print(f"  bio: {(r['bio'] or '')[:200]}")
        console.print(f"  photos: {cfg.photos_path / r['profile_id']}")
        ans = typer.prompt("  your call [l]ike / [n]ope / [s]kip", default="s").lower()[:1]
        if ans in ("l", "n"):
            st.relabel(r["profile_id"], 1 if ans == "l" else 0)


@app.command()
def stats(config: str = _cfg_opt):
    """Show what is stored locally."""
    cfg = _cfg(config)
    st = _storage(cfg)
    day = time.time() - (time.time() % 86400)
    t = Table(title=str(cfg.db_path))
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("profiles", str(st.conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]))
    t.add_row("photos", str(st.conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]))
    t.add_row("embeddings", str(st.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]))
    t.add_row("decisions (total)", str(st.count_decisions()))
    t.add_row("decisions today (auto)", str(st.count_decisions(since_ts=day, source="auto")))
    t.add_row("likes / nopes", f"{st.count_decisions(action='like')} / {st.count_decisions(action='nope')}")
    t.add_row("manual labels", str(st.count_decisions(source="manual")))
    t.add_row("captchas today", str(st.count_events("captcha", day)))
    for k, v in st.conn.execute("SELECT label||'/'||kind, COUNT(*) FROM reference_vectors GROUP BY 1"):
        t.add_row(f"references {k}", str(v))
    console.print(t)


@app.command()
def export(out: Path = typer.Argument(Path("decisions.json")), config: str = _cfg_opt):
    """Dump decisions + profile metadata to JSON (stays on your disk)."""
    cfg = _cfg(config)
    st = _storage(cfg)
    rows = st.conn.execute(
        "SELECT d.*, p.name, p.age, p.bio FROM decisions d JOIN profiles p ON p.id=d.profile_id ORDER BY d.ts"
    ).fetchall()
    out.write_text(json.dumps([dict(r) for r in rows], ensure_ascii=False, indent=1))
    console.print(f"wrote {len(rows)} rows to {out}")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
