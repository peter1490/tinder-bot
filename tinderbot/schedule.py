"""Day-level rhythm for the unattended bot (``tinderbot auto``).

A human does not open Tinder at 09:00 sharp every day, swipe exactly three times 40-90 cards and
stop.  They open it a random number of times, mostly in the evening, sometimes at lunch, some days
not at all, for a random amount of time, and they close the app in between.  This module produces
that rhythm:

* :func:`plan_day` draws a :class:`DayPlan` for one calendar day: possibly a rest day, otherwise
  1-N sessions at weighted random times of day, each with a right-skewed swipe count, the whole day
  bounded by a randomised budget and a ramp-up factor for freshly automated accounts.
* :class:`Scheduler` persists the plan (restarts keep the same plan), sleeps with the browser
  **closed** until the next slot, runs one session through :meth:`Runner.run_session`, and applies
  the safety policy to the outcome: cancel the rest of the day, pause for hours, or halt until a
  human runs ``tinderbot resume``.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .browser.humanize import lognormal_delay
from .browser.pacing import local_midnight_ts
from .config import Config
from .storage import Storage

# Rough duration estimate used only for spacing sessions inside the day (seconds per swipe).
SECONDS_PER_SWIPE = 9.0
WARMUP_SECONDS = 20.0

META_HALT = "halt"
META_PAUSE_UNTIL = "pause_until"
META_UNSOLVED_STREAK = "unsolved_challenge_streak"
META_ERROR_STREAK = "session_error_streak"

SLOT_PENDING = "pending"
SLOT_RUNNING = "running"
SLOT_DONE = "done"
SLOT_MISSED = "missed"        # process was asleep past the grace window
SLOT_CANCELLED = "cancelled"  # rest of the day dropped (out of likes, challenge, budget)
SLOT_PAUSED = "paused"        # fell inside a pause window


@dataclass
class Slot:
    start: float                 # unix timestamp
    swipes: int
    status: str = SLOT_PENDING
    done: int = 0
    reason: str = ""

    @property
    def start_dt(self) -> dt.datetime:
        return dt.datetime.fromtimestamp(self.start)

    def estimated_seconds(self) -> float:
        return WARMUP_SECONDS + self.swipes * SECONDS_PER_SWIPE


@dataclass
class DayPlan:
    date: str                    # YYYY-MM-DD (local)
    rest: bool = False
    budget: int = 0
    ramp: float = 1.0
    slots: list[Slot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DayPlan:
        return cls(date=d["date"], rest=bool(d.get("rest")), budget=int(d.get("budget", 0)),
                   ramp=float(d.get("ramp", 1.0)), slots=[Slot(**s) for s in d.get("slots", [])])

    def pending(self) -> list[Slot]:
        return [s for s in self.slots if s.status == SLOT_PENDING]

    def next_pending(self) -> Slot | None:
        p = self.pending()
        return min(p, key=lambda s: s.start) if p else None

    def cancel_pending(self, reason: str) -> int:
        n = 0
        for s in self.pending():
            s.status, s.reason = SLOT_CANCELLED, reason
            n += 1
        return n

    def planned_swipes(self) -> int:
        return sum(s.swipes for s in self.slots)

    def describe(self) -> list[str]:
        if self.rest:
            return [f"{self.date}: rest day"]
        lines = [f"{self.date}: {len(self.slots)} session(s), {self.planned_swipes()} swipes planned "
                 f"(budget {self.budget}, ramp x{self.ramp:.2f})"]
        for s in sorted(self.slots, key=lambda s: s.start):
            extra = f"  [{s.status}{(' ' + s.reason) if s.reason else ''}{(' done=' + str(s.done)) if s.done else ''}]"
            lines.append(f"  {s.start_dt:%H:%M}  {s.swipes:3d} swipes  ~{s.estimated_seconds() / 60:.0f} min{extra}")
        return lines


def _stochastic_round(x: float, rng: random.Random) -> int:
    f = math.floor(x)
    return f + (1 if rng.random() < x - f else 0)


def ramp_factor(cfg: Config, storage: Storage, now: float | None = None) -> float:
    """Scale the budget up over the first ``ramp_days`` of automation (new accounts get watched)."""
    sc = cfg.schedule
    if sc.ramp_days <= 0:
        return 1.0
    first = storage.first_decision_ts("auto")
    if first is None:
        return sc.ramp_start
    days = max(0.0, ((now if now is not None else time.time()) - first) / 86400.0)
    return sc.ramp_start + (1.0 - sc.ramp_start) * min(1.0, days / sc.ramp_days)


def plan_day(cfg: Config, day: dt.date, rng: random.Random, ramp: float = 1.0) -> DayPlan:
    """Draw one day's plan.  Pure function of (config, day, rng, ramp): unit-testable and reproducible."""
    pc, sc = cfg.pacing, cfg.schedule
    plan = DayPlan(date=day.isoformat(), ramp=ramp)
    if rng.random() < sc.p_rest_day:
        plan.rest = True
        return plan

    lo_h, hi_h = pc.active_hours
    day_start = dt.datetime(day.year, day.month, day.day)
    window_start = (day_start + dt.timedelta(hours=lo_h)).timestamp()
    window_end = (day_start + dt.timedelta(hours=hi_h)).timestamp()

    plan.budget = max(0, int(round(pc.max_swipes_per_day * rng.uniform(*sc.day_budget_jitter) * ramp)))
    lo_s, hi_s = pc.sessions_per_day
    n = rng.randint(lo_s, hi_s)
    if day.weekday() >= 5 and sc.weekend_multiplier != 1.0:
        n = _stochastic_round(n * sc.weekend_multiplier, rng)
    n = max(lo_s, min(n, hi_s + 1))
    if n <= 0 or plan.budget <= 0:
        return plan

    # swipe counts: right-skewed, then scaled into the day's budget
    lo_w, hi_w = pc.swipes_per_session
    counts = [max(1, int(round(lognormal_delay(lo_w, hi_w, rng)))) for _ in range(n)]
    total = sum(counts)
    if total > plan.budget:
        counts = [int(c * plan.budget / total) for c in counts]
    counts = [c for c in counts if c >= max(5, lo_w // 3)]
    if not counts:
        counts = [min(plan.budget, max(5, lo_w))]

    # start times: weighted by hour, must fit before the end of the active window, spaced by min_gap
    hours = [h for h in range(lo_h, hi_h) if sc.hour_weights[h] > 0]
    weights = [sc.hour_weights[h] for h in hours]
    if not hours:
        hours, weights = list(range(lo_h, hi_h)), [1.0] * (hi_h - lo_h)
    gap = sc.min_gap_minutes * 60.0
    best: list[Slot] = []
    for _attempt in range(60):
        slots: list[Slot] = []
        for c in counts:
            est = WARMUP_SECONDS + c * SECONDS_PER_SWIPE
            for _try in range(20):
                h = rng.choices(hours, weights)[0]
                start = (day_start + dt.timedelta(hours=h, minutes=rng.uniform(0, 60))).timestamp()
                if window_start <= start and start + est <= window_end:
                    slots.append(Slot(start=start, swipes=c))
                    break
        slots.sort(key=lambda s: s.start)
        ok = all(slots[i + 1].start >= slots[i].start + slots[i].estimated_seconds() + gap
                 for i in range(len(slots) - 1))
        if ok and len(slots) == len(counts):
            best = slots
            break
        if len(slots) > len(best):
            best = slots
    # drop overlapping slots if the rejection sampling did not converge (tiny windows / many sessions)
    cleaned: list[Slot] = []
    for s in sorted(best, key=lambda s: s.start):
        if not cleaned or s.start >= cleaned[-1].start + cleaned[-1].estimated_seconds() + gap:
            cleaned.append(s)
    plan.slots = cleaned
    return plan


class Scheduler:
    """Run planned sessions day after day, unattended, with the browser closed in between."""

    def __init__(self, cfg: Config, storage: Storage, runner_factory: Callable[[], Any],
                 rng: random.Random | None = None, now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.storage = storage
        self._runner_factory = runner_factory
        self._runner = None
        self.rng = rng or random.Random()
        self.now = now
        self.sleep = sleep
        self.log = log
        self.sessions_run = 0

    # ---- persisted state -----------------------------------------------------------------
    @property
    def runner(self):
        if self._runner is None:
            self._runner = self._runner_factory()
        return self._runner

    def halted(self) -> dict[str, Any] | None:
        return self.storage.get_meta(META_HALT)

    def halt(self, reason: str, detail: str = "") -> None:
        self.storage.set_meta(META_HALT, {"reason": reason, "detail": detail, "ts": self.now()})
        self.storage.log_event("halt", {"reason": reason, "detail": detail})
        self.log(f"HALTED: {reason} {detail}".rstrip() + "  -> fix the cause, then run `tinderbot resume`.")

    def resume(self) -> None:
        for k in (META_HALT, META_PAUSE_UNTIL, META_UNSOLVED_STREAK, META_ERROR_STREAK):
            self.storage.delete_meta(k)
        self.storage.log_event("resume", {})

    def pause_until(self) -> float:
        return float(self.storage.get_meta(META_PAUSE_UNTIL, 0.0) or 0.0)

    def pause_for(self, seconds: float, reason: str) -> None:
        until = self.now() + seconds
        self.storage.set_meta(META_PAUSE_UNTIL, until)
        self.storage.log_event("pause", {"reason": reason, "until": until})
        self.log(f"Paused ({reason}) until {dt.datetime.fromtimestamp(until):%Y-%m-%d %H:%M}.")

    def _streak(self, key: str, delta: int | None) -> int:
        """Increment (delta=1) or reset (delta=None) a persisted counter; returns the new value."""
        v = 0 if delta is None else int(self.storage.get_meta(key, 0) or 0) + delta
        self.storage.set_meta(key, v)
        return v

    # ---- plans ---------------------------------------------------------------------------
    @staticmethod
    def _key(day: dt.date) -> str:
        return f"dayplan:{day.isoformat()}"

    def load_plan(self, day: dt.date) -> DayPlan | None:
        d = self.storage.get_meta(self._key(day))
        return DayPlan.from_dict(d) if d else None

    def save_plan(self, plan: DayPlan) -> None:
        self.storage.set_meta(f"dayplan:{plan.date}", plan.to_dict())

    def plan_for(self, day: dt.date, persist: bool = True) -> DayPlan:
        plan = self.load_plan(day)
        if plan is None:
            plan = plan_day(self.cfg, day, self.rng, ramp=ramp_factor(self.cfg, self.storage, self.now()))
            if persist:
                self.save_plan(plan)
                self.storage.log_event("day_planned", {"date": plan.date, "rest": plan.rest,
                                                       "sessions": len(plan.slots), "swipes": plan.planned_swipes()})
                for line in plan.describe():
                    self.log(line)
        return plan

    def today(self) -> dt.date:
        return dt.datetime.fromtimestamp(self.now()).date()

    # ---- one scheduler step ---------------------------------------------------------------
    def tick(self) -> tuple[str, float]:
        """Do at most one thing and return ``(what, seconds_to_sleep)``.

        ``what`` is one of: ``halted``, ``rest``, ``day_done``, ``wait``, ``missed``, ``paused``, ``ran``.
        """
        if self.halted():
            return "halted", 0.0
        now = self.now()
        plan = self.plan_for(self.today())
        slot = plan.next_pending()
        if slot is None:
            # nothing (more) today: sleep until a few minutes after local midnight
            next_midnight = local_midnight_ts(now) + 86400.0
            return ("rest" if plan.rest else "day_done"), max(60.0, next_midnight + self.rng.uniform(60, 600) - now)
        if now < slot.start:
            return "wait", min(self.cfg.schedule.poll_seconds, slot.start - now)
        if now > slot.start + self.cfg.schedule.slot_grace_minutes * 60.0:
            slot.status, slot.reason = SLOT_MISSED, "late"
            self.save_plan(plan)
            self.log(f"Missed the {slot.start_dt:%H:%M} slot (process was asleep); skipping it.")
            return "missed", 0.0
        pause = self.pause_until()
        if now < pause:
            slot.status, slot.reason = SLOT_PAUSED, "pause"
            self.save_plan(plan)
            return "paused", 0.0
        self._run_slot(plan, slot)
        return "ran", 0.0

    def _run_slot(self, plan: DayPlan, slot: Slot) -> None:
        sc = self.cfg.schedule
        slot.status = SLOT_RUNNING
        self.save_plan(plan)
        self.log(f"Session {slot.start_dt:%H:%M}: {slot.swipes} swipes planned")
        result = self.runner.run_session(slot.swipes)
        self.sessions_run += 1
        slot.status, slot.done, slot.reason = SLOT_DONE, result.done, result.reason
        self.log(f"Session over: {result.done}/{result.planned} swipes, liked={result.stats.liked} "
                 f"noped={result.stats.noped} downgraded={result.stats.downgraded} ({result.reason})")

        reason = result.reason
        if reason in ("planned", "ended_early"):
            self._streak(META_ERROR_STREAK, None)
        if reason in ("planned", "ended_early", "no_card", "swipe_unconfirmed"):
            self.save_plan(plan)
            return
        if reason in ("account_notice", "needs_login"):
            plan.cancel_pending(reason)
            self.save_plan(plan)
            self.halt(reason, result.error or "check the browser window / run `tinderbot login`")
            return
        if reason == "error":
            n = self._streak(META_ERROR_STREAK, 1)
            if n >= sc.max_consecutive_errors:
                plan.cancel_pending("error")
                self.save_plan(plan)
                self.halt("errors", f"{n} consecutive session errors; last: {result.error}")
                return
            # try again at the next slot; a single failure is usually a flaky launch or network
            self.save_plan(plan)
            return
        if reason == "captcha_unsolved":
            plan.cancel_pending(reason)
            self.save_plan(plan)
            n = self._streak(META_UNSOLVED_STREAK, 1)
            if n >= sc.max_unsolved_challenges:
                self.halt("unsolved_challenges", f"{n} challenges in a row nobody solved")
                return
            lo, hi = sc.unsolved_challenge_pause_hours
            self.pause_for(self.rng.uniform(lo, hi) * 3600.0, "unsolved challenge")
            return
        if reason == "captcha_solved":
            self._streak(META_UNSOLVED_STREAK, None)
            lo, hi = self.cfg.captcha.cooldown_after_captcha_minutes
            self.pause_for(self.rng.uniform(lo, hi) * 60.0, "challenge solved, cooling down")
            self.save_plan(plan)
            return
        # out_of_likes, captcha_limit, budget: the day is over
        plan.cancel_pending(reason)
        self.save_plan(plan)

    # ---- main loop ------------------------------------------------------------------------
    def run(self, max_sessions: int | None = None, max_days: int | None = None) -> str:
        """Loop until halted (or the optional limits are reached).  Returns the last ``what``."""
        days_seen: set[str] = set()
        what = "start"
        while True:
            today = self.today().isoformat()
            if max_days is not None and today not in days_seen and len(days_seen) >= max_days:
                return what
            days_seen.add(today)
            what, delay = self.tick()
            if what == "halted":
                h = self.halted() or {}
                self.log(f"Bot is halted ({h.get('reason')}: {h.get('detail')}). Run `tinderbot resume` to continue.")
                return what
            if max_sessions is not None and self.sessions_run >= max_sessions:
                return what
            if delay > 0:
                self.sleep(delay)
