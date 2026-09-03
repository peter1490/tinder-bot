"""Session pacing: daily/session budgets, active hours, breaks, and per-profile dwell time.

The numbers are deliberately conservative: Tinder's own risk model watches swipe cadence and volume
(free accounts are capped at ~100 likes/day anyway), and account bans are far costlier than speed.
"""

from __future__ import annotations

import datetime as dt
import random
import time
from dataclasses import dataclass

from ..config import PacingConfig
from ..storage import Storage
from .humanize import ReadingModel, lognormal_delay


def local_midnight_ts() -> float:
    now = dt.datetime.now()
    return dt.datetime(now.year, now.month, now.day).timestamp()


@dataclass
class SessionPlan:
    swipes: int
    started: float


class Pacer:
    def __init__(self, cfg: PacingConfig, storage: Storage, rng: random.Random | None = None):
        self.cfg = cfg
        self.storage = storage
        self.rng = rng or random.Random()
        self.reading = ReadingModel(cfg.base_seconds, cfg.per_photo_seconds, cfg.per_bio_char_seconds)
        self.slowdown = 1.0
        self.session: SessionPlan | None = None

    # ---- budgets -------------------------------------------------------------------
    def swipes_today(self) -> int:
        return self.storage.count_decisions(since_ts=local_midnight_ts(), source="auto")

    def sessions_today(self) -> int:
        return self.storage.count_events("session_start", local_midnight_ts())

    def remaining_today(self) -> int:
        return max(0, self.cfg.max_swipes_per_day - self.swipes_today())

    def in_active_hours(self, now: dt.datetime | None = None) -> bool:
        now = now or dt.datetime.now()
        lo, hi = self.cfg.active_hours
        return lo <= now.hour < hi

    def seconds_until_active(self, now: dt.datetime | None = None) -> float:
        now = now or dt.datetime.now()
        if self.in_active_hours(now):
            return 0.0
        lo = self.cfg.active_hours[0]
        target = now.replace(hour=lo, minute=self.rng.randint(0, 40), second=0, microsecond=0)
        if target <= now:
            target += dt.timedelta(days=1)
        return (target - now).total_seconds()

    def start_session(self, max_override: int | None = None) -> SessionPlan | None:
        if self.sessions_today() >= self.cfg.sessions_per_day:
            return None
        remaining = self.remaining_today()
        if remaining <= 0:
            return None
        n = self.rng.randint(*self.cfg.swipes_per_session)
        n = min(n, remaining)
        if max_override:
            n = min(n, max_override)
        self.session = SessionPlan(swipes=n, started=time.time())
        self.storage.log_event("session_start", {"planned": n})
        return self.session

    def end_session(self, done: int) -> None:
        self.storage.log_event("session_end", {"done": done})
        self.session = None

    def break_seconds(self) -> float:
        lo, hi = self.cfg.break_between_sessions_min
        return self.rng.uniform(lo, hi) * 60 * self.slowdown

    # ---- per profile ------------------------------------------------------------------
    def plan_profile(self, photo_count: int, bio_len: int) -> dict:
        browse = 0
        if photo_count > 1 and self.rng.random() < self.cfg.p_browse_photos:
            browse = self.rng.randint(1, min(self.cfg.max_photos_browsed, photo_count - 1))
        return {
            "browse_photos": browse,
            "open_profile": self.rng.random() < self.cfg.p_open_profile,
            "read_seconds": self.reading.seconds(browse, bio_len, self.rng) * self.slowdown,
            "micro_break": lognormal_delay(*self.cfg.micro_break_seconds, self.rng)
            if self.rng.random() < self.cfg.p_micro_break else 0.0,
        }

    def post_action_delay(self) -> float:
        return lognormal_delay(0.4, 1.6, self.rng) * self.slowdown
