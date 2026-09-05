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


def _as_pair(v: Any) -> tuple:
    if isinstance(v, (int, float)):
        return (v, v)
    v = tuple(v)
    if len(v) != 2 or v[0] > v[1]:
        raise ValueError(f"expected [low, high] with low <= high, got {v}")
    return v


class PacingConfig(BaseModel):
    swipes_per_session: tuple[int, int] = (15, 80)
    # [min, max] sessions per day; the scheduler draws a different count every day (a plain int is
    # accepted and means "exactly that many", which is what the legacy ``swipe --loop`` mode does).
    sessions_per_day: tuple[int, int] = (1, 4)
    max_swipes_per_day: int = 160
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
    # Verdict-aware dwell: people linger on profiles they like and flick past the ones they don't.
    like_dwell_multiplier: tuple[float, float] = (1.1, 1.8)
    nope_dwell_multiplier: tuple[float, float] = (0.5, 1.0)
    # Per-session "persona": overall tempo and how often the keyboard shortcuts are used instead of
    # the mouse.  Drawn once per session so a session is internally consistent but sessions differ.
    session_tempo: tuple[float, float] = (0.8, 1.35)
    keyboard_preference: tuple[float, float] = (0.1, 0.8)
    # Orientation pause after the recs screen appears, before the first swipe.
    session_warmup_seconds: tuple[float, float] = (3.0, 12.0)
    # Small chance to quit a session early ("got bored"), evaluated per swipe after half the plan.
    p_end_session_early: float = 0.01
    # Like-rate governor: Tinder's risk model treats right-swiping (almost) everyone as spam.  When
    # the like share of the last ``like_ratio_window`` auto decisions exceeds ``max_like_ratio`` the
    # weakest LIKE verdicts are turned into NOPEs until the ratio recovers.  1.0 disables it.
    max_like_ratio: float = 0.55
    like_ratio_window: int = 50
    like_ratio_min_samples: int = 12

    @field_validator("swipes_per_session", "break_between_sessions_min", "base_seconds",
                     "per_photo_seconds", "micro_break_seconds", "active_hours", "sessions_per_day",
                     "like_dwell_multiplier", "nope_dwell_multiplier", "session_tempo",
                     "keyboard_preference", "session_warmup_seconds", mode="before")
    @classmethod
    def _pair(cls, v: Any) -> tuple:
        return _as_pair(v)

    @field_validator("max_like_ratio")
    @classmethod
    def _ratio(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("max_like_ratio must be in (0, 1]")
        return v


class ScheduleConfig(BaseModel):
    """Day-level rhythm used by ``tinderbot auto`` (fully unattended mode).

    Every day gets its own plan: maybe a rest day, otherwise a random number of sessions placed at
    random times of day (weighted by ``hour_weights``), each with a random, right-skewed swipe count.
    The plan is persisted so restarts do not re-roll it.
    """

    p_rest_day: float = 0.12
    # Relative likelihood of a session starting in each hour of the day (index 0 = midnight).  Hours
    # outside ``pacing.active_hours`` are ignored regardless of their weight.  Default: a small
    # morning bump, a lunch bump and a strong evening peak.
    hour_weights: list[float] = Field(default_factory=lambda: [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 0.8, 0.7, 0.7,
        1.2, 1.1, 0.6, 0.6, 0.7, 0.9, 1.2, 1.5, 2.0, 2.2, 1.8, 0.9,
    ])
    # Weekend days get this factor on the session count (>1 = more swiping at the weekend).
    weekend_multiplier: float = 1.15
    min_gap_minutes: float = 45.0
    # If the process wakes up later than this after a slot's start (laptop asleep, reboot) the slot
    # is skipped rather than run late; a human would just have missed that moment.
    slot_grace_minutes: float = 40.0
    # The day's swipe target is ``pacing.max_swipes_per_day`` times a factor drawn from this range.
    day_budget_jitter: tuple[float, float] = (0.5, 1.0)
    # Warm-up: for the first ``ramp_days`` of automation the budget is scaled from ``ramp_start`` to 1.
    ramp_days: int = 7
    ramp_start: float = 0.4
    # Unattended safety: a challenge nobody solved cancels the rest of the day and pauses the bot for
    # this many hours; after ``max_unsolved_challenges`` consecutive unsolved ones it halts until
    # ``tinderbot resume``.  Any account-level notice or a lost login halts immediately.
    unsolved_challenge_pause_hours: tuple[float, float] = (12, 24)
    max_unsolved_challenges: int = 2
    max_consecutive_errors: int = 3
    # Seconds between wall-clock checks while idle (the browser is closed while waiting).
    poll_seconds: float = 30.0

    @field_validator("day_budget_jitter", "unsolved_challenge_pause_hours", mode="before")
    @classmethod
    def _pair(cls, v: Any) -> tuple:
        return _as_pair(v)

    @field_validator("hour_weights")
    @classmethod
    def _hours(cls, v: list[float]) -> list[float]:
        if len(v) != 24 or any(w < 0 for w in v):
            raise ValueError("hour_weights needs 24 non-negative numbers")
        return v

    @field_validator("p_rest_day", "ramp_start")
    @classmethod
    def _unit(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("expected a value in [0, 1]")
        return v


class CaptchaConfig(BaseModel):
    wait_for_human_max_minutes: float = 10
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


class CrushConfig(BaseModel):
    """Super Likes for "crushes" and a Super Like note for "super crushes", both rationed per day."""

    enabled: bool = True
    super_like_threshold: float = 0.9      # final probability above which a profile is a crush
    message_threshold: float = 0.95        # ... and a super crush (Super Like sent with a note)
    max_super_likes_per_day: int = 1
    max_messages_per_day: int = 1
    require_learned: bool = True           # never spend a Super Like on the hand-tuned prior alone
    messages: list[str] = Field(default_factory=lambda: [
        "Hey {name}! Your profile really stood out, I would love to hear more about you.",
        "Hi {name} :) Something about your photos made me smile. Coffee sometime?",
    ])

    @field_validator("messages")
    @classmethod
    def _short_notes(cls, v: list[str]) -> list[str]:
        for m in v:
            if len(m) > 140:
                raise ValueError("Super Like notes are limited to 140 characters")
        return v


class ModelsConfig(BaseModel):
    face_detector: str = "buffalo_l/det_10g"
    face_recognizer: str = "buffalo_l/w600k_r50"
    clip: str = "clip-vit-base-patch32"
    threads: int = 0


class Config(BaseModel):
    data_dir: str = "data"
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    pacing: PacingConfig = Field(default_factory=PacingConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    captcha: CaptchaConfig = Field(default_factory=CaptchaConfig)
    likeness: LikenessConfig = Field(default_factory=LikenessConfig)
    crush: CrushConfig = Field(default_factory=CrushConfig)
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
