"""Session pacing: daily/session budgets, active hours, breaks, per-profile dwell time, like-rate cap.

The numbers are deliberately conservative: Tinder's own risk model watches swipe cadence, volume and
the like share (free accounts are capped at ~100 likes/day anyway), and account bans are far costlier
than speed.  Day-level rhythm (when sessions happen, how many, rest days) lives in
:mod:`tinderbot.schedule`; this module handles what happens inside a session.
"""

from __future__ import annotations

import datetime as dt
import random
import time
from dataclasses import dataclass

from ..config import PacingConfig
from ..storage import Storage
from .humanize import ReadingModel, lognormal_delay

# Sources that count towards the daily swipe budget: what the bot did and what you did in shadow mode.
BUDGET_SOURCES = ("auto", "manual")


def local_midnight_ts(now: float | None = None) -> float:
    d = dt.datetime.fromtimestamp(now if now is not None else time.time())
    return dt.datetime(d.year, d.month, d.day).timestamp()


@dataclass
class SessionPlan:
    swipes: int
    started: float


@dataclass
class Persona:
    """Per-session behavioural constants, drawn once so a session is internally consistent."""

    tempo: float = 1.0               # multiplies every dwell/pause (0.8 = a quicker session)
    keyboard_pref: float = 0.0       # probability of using keyboard shortcuts instead of the mouse
    browse_bias: float = 1.0         # multiplies p_browse_photos

    @classmethod
    def draw(cls, cfg: PacingConfig, rng: random.Random) -> Persona:
        return cls(
            tempo=rng.uniform(*cfg.session_tempo),
            keyboard_pref=rng.uniform(*cfg.keyboard_preference),
            browse_bias=rng.uniform(0.7, 1.25),
        )


class LikeGovernor:
    """Keep the share of LIKEs among recent auto decisions below ``max_like_ratio``.

    Right-swiping (almost) everyone is the single most reported cause of shadow-bans and Arkose
    challenges.  The scorer decides on taste; this class decides whether acting on a LIKE now would
    push the recent like share over the cap, in which case the swipe is turned into a NOPE.  The
    training label still records the scorer's verdict, so the learned model is not polluted.
    """

    def __init__(self, cfg: PacingConfig, storage: Storage):
        self.cfg = cfg
        self.storage = storage
        self.downgraded = 0

    def recent_ratio(self) -> tuple[float, int]:
        actions = self.storage.recent_actions(self.cfg.like_ratio_window, source="auto")
        n = len(actions)
        likes = sum(1 for a in actions if a == "like")
        return (likes / n if n else 0.0), n

    def allow_like(self) -> bool:
        if self.cfg.max_like_ratio >= 1.0:
            return True
        actions = self.storage.recent_actions(self.cfg.like_ratio_window, source="auto")
        n = len(actions)
        if n < self.cfg.like_ratio_min_samples:
            return True
        likes = sum(1 for a in actions if a == "like")
        # ratio *after* this like, over a window that drops the oldest entry when full
        if n >= self.cfg.like_ratio_window:
            likes -= 1 if actions[-1] == "like" else 0
            n -= 1
        projected = (likes + 1) / (n + 1)
        if projected > self.cfg.max_like_ratio:
            self.downgraded += 1
            return False
        return True


class Pacer:
    def __init__(self, cfg: PacingConfig, storage: Storage, rng: random.Random | None = None):
        self.cfg = cfg
        self.storage = storage
        self.rng = rng or random.Random()
        self.reading = ReadingModel(cfg.base_seconds, cfg.per_photo_seconds, cfg.per_bio_char_seconds)
        self.slowdown = 1.0
        self.session: SessionPlan | None = None
        self.persona = Persona()

    # ---- budgets -------------------------------------------------------------------
    def swipes_today(self) -> int:
        return self.storage.count_decisions(since_ts=local_midnight_ts(), source=BUDGET_SOURCES)

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

    def draw_session_swipes(self) -> int:
        """Right-skewed draw in ``swipes_per_session``: many short sessions, a few long ones."""
        lo, hi = self.cfg.swipes_per_session
        return int(round(lognormal_delay(lo, hi, self.rng)))

    def start_session(self, max_override: int | None = None) -> SessionPlan | None:
        """Legacy ``swipe --loop`` entry point: fixed session count per day, random size."""
        if self.sessions_today() >= self.cfg.sessions_per_day[1]:
            return None
        remaining = self.remaining_today()
        if remaining <= 0:
            return None
        n = min(self.draw_session_swipes(), remaining)
        if max_override:
            n = min(n, max_override)
        return self.begin_session(n)

    def begin_session(self, n: int) -> SessionPlan | None:
        """Start a session of ``n`` swipes (already decided by the scheduler), bounded by the budget."""
        n = min(int(n), self.remaining_today())
        if n <= 0:
            return None
        self.persona = Persona.draw(self.cfg, self.rng)
        self.session = SessionPlan(swipes=n, started=time.time())
        self.storage.log_event("session_start", {"planned": n, "tempo": round(self.persona.tempo, 2),
                                                 "keyboard_pref": round(self.persona.keyboard_pref, 2)})
        return self.session

    def end_session(self, done: int, reason: str = "planned") -> None:
        self.storage.log_event("session_end", {"done": done, "reason": reason})
        self.session = None

    def break_seconds(self) -> float:
        lo, hi = self.cfg.break_between_sessions_min
        return self.rng.uniform(lo, hi) * 60 * self.slowdown

    def warmup_seconds(self) -> float:
        return lognormal_delay(*self.cfg.session_warmup_seconds, self.rng) * self.persona.tempo

    def should_end_early(self, done: int, planned: int) -> bool:
        """Occasionally quit a session before the planned count (people get bored / interrupted)."""
        if planned <= 0 or done < planned / 2:
            return False
        return self.rng.random() < self.cfg.p_end_session_early

    # ---- per profile ------------------------------------------------------------------
    def plan_profile(self, photo_count: int, bio_len: int, like: bool | None = None) -> dict:
        """Dwell plan for one card.  ``like`` (the verdict) lengthens or shortens the look."""
        p_browse = min(1.0, self.cfg.p_browse_photos * self.persona.browse_bias)
        mult = self.persona.tempo * self.slowdown
        if like is True:
            mult *= self.rng.uniform(*self.cfg.like_dwell_multiplier)
            p_browse = min(1.0, p_browse * 1.25)
        elif like is False:
            mult *= self.rng.uniform(*self.cfg.nope_dwell_multiplier)
            p_browse *= 0.7
        browse = 0
        if photo_count > 1 and self.rng.random() < p_browse:
            browse = self.rng.randint(1, min(self.cfg.max_photos_browsed, photo_count - 1))
        return {
            "browse_photos": browse,
            "open_profile": self.rng.random() < self.cfg.p_open_profile * {True: 1.3, False: 0.7}.get(like, 1.0),
            "read_seconds": self.reading.seconds(browse, bio_len, self.rng) * mult,
            "micro_break": lognormal_delay(*self.cfg.micro_break_seconds, self.rng)
            if self.rng.random() < self.cfg.p_micro_break else 0.0,
        }

    def post_action_delay(self) -> float:
        return lognormal_delay(0.4, 1.6, self.rng) * self.slowdown * self.persona.tempo
