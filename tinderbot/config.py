"""Typed configuration loaded from ``config.toml`` (falls back to ``config.example.toml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class BrowserConfig(BaseModel):
    channel: str = "chrome"
    executable_path: str = ""
    headless: bool = False
    locale: str = ""
    slow_mo_ms: int = 0


class PacingConfig(BaseModel):
    swipes_per_session: tuple[int, int] = (40, 90)
    sessions_per_day: int = 3
    max_swipes_per_day: int = 200
    break_between_sessions_min: tuple[float, float] = (20, 75)
    active_hours: tuple[int, int] = (9, 23)
    base_seconds: tuple[float, float] = (1.2, 4.5)
    per_photo_seconds: tuple[float, float] = (0.8, 2.5)
    per_bio_char_seconds: float = 0.012
    p_browse_photos: float = 0.65
    max_photos_browsed: int = 4
    p_open_profile: float = 0.15
    p_micro_break: float = 0.06
    micro_break_seconds: tuple[float, float] = (15, 90)

    @field_validator("swipes_per_session", "break_between_sessions_min", "base_seconds",
                     "per_photo_seconds", "micro_break_seconds", "active_hours", mode="before")
    @classmethod
    def _pair(cls, v: Any) -> tuple:
        if isinstance(v, (int, float)):
            return (v, v)
        v = tuple(v)
        if len(v) != 2 or v[0] > v[1]:
            raise ValueError(f"expected [low, high] with low <= high, got {v}")
        return v


class CaptchaConfig(BaseModel):
    wait_for_human_max_minutes: float = 30
    cooldown_after_captcha_minutes: tuple[float, float] = (30, 90)
    max_captchas_per_day: int = 2
    notify_desktop: bool = True
    sound: bool = True

    @field_validator("cooldown_after_captcha_minutes", mode="before")
    @classmethod
    def _pair(cls, v: Any) -> tuple:
        return (v, v) if isinstance(v, (int, float)) else tuple(v)


class Weights(BaseModel):
    face_sim_liked: float = 3.0
    face_sim_disliked: float = -2.0
    clip_sim_liked: float = 2.0
    clip_sim_disliked: float = -1.5
    text_prompts: float = 1.0
    photo_quality: float = 0.5
    bio_keywords: float = 0.5
    bias: float = -1.0


class Prompts(BaseModel):
    positive: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)


class Learning(BaseModel):
    min_examples: int = 40
    blend_full_at: int = 300
    retrain_every: int = 25


class LikenessConfig(BaseModel):
    min_age: int = 18
    max_age: int = 99
    max_distance_km: float = 0
    require_face_photo: bool = True
    require_verified: bool = False
    blocked_bio_keywords: list[str] = Field(default_factory=list)
    liked_bio_keywords: list[str] = Field(default_factory=list)
    like_threshold: float = 0.55
    uncertain_band: float = 0.08
    weights: Weights = Field(default_factory=Weights)
    prompts: Prompts = Field(default_factory=Prompts)
    learning: Learning = Field(default_factory=Learning)


class ModelsConfig(BaseModel):
    face_detector: str = "buffalo_l/det_10g"
    face_recognizer: str = "buffalo_l/w600k_r50"
    clip: str = "clip-vit-base-patch32"
    threads: int = 0


class Config(BaseModel):
    data_dir: str = "data"
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    pacing: PacingConfig = Field(default_factory=PacingConfig)
    captcha: CaptchaConfig = Field(default_factory=CaptchaConfig)
    likeness: LikenessConfig = Field(default_factory=LikenessConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    # ---- derived paths -------------------------------------------------
    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def db_path(self) -> Path:
        return self.data_path / "tinderbot.db"

    @property
    def photos_path(self) -> Path:
        return self.data_path / "photos"

    @property
    def models_path(self) -> Path:
        return self.data_path / "models"

    @property
    def profile_path(self) -> Path:
        return self.data_path / "browser-profile"

    def ensure_dirs(self) -> None:
        for p in (self.data_path, self.photos_path, self.models_path, self.profile_path):
            p.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    """Load config.toml (or the example file when no user config exists)."""
    candidates = [Path(path)] if path else [PROJECT_ROOT / "config.toml", PROJECT_ROOT / "config.example.toml"]
    for c in candidates:
        if c.exists():
            with open(c, "rb") as f:
                return Config.model_validate(tomllib.load(f))
    return Config()
