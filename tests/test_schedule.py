"""Day planner and unattended scheduler (no browser: the runner is faked, time is simulated)."""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass

import pytest

from tinderbot.config import Config, PacingConfig, ScheduleConfig
from tinderbot.runner import RunStats, SessionResult
from tinderbot.schedule import (
    SLOT_CANCELLED,
    SLOT_DONE,
    SLOT_MISSED,
    DayPlan,
    Scheduler,
    plan_day,
    ramp_factor,
)
from tinderbot.storage import ProfileRecord


def _cfg(**schedule) -> Config:
    return Config(pacing=PacingConfig(active_hours=(9, 23), sessions_per_day=(1, 4), swipes_per_session=(15, 80),
                                      max_swipes_per_day=160),
                  schedule=ScheduleConfig(**schedule))


DAY = dt.date(2026, 9, 7)  # a Monday


def test_plan_day_is_random_but_within_bounds():
    cfg = _cfg(p_rest_day=0.0, ramp_days=0)
    counts, starts = set(), set()
    for seed in range(60):
        plan = plan_day(cfg, DAY, random.Random(seed))
        assert not plan.rest
        assert 1 <= len(plan.slots) <= 5
        assert plan.planned_swipes() <= plan.budget <= 160
        assert plan.budget >= 80
        for s in plan.slots:
            assert 9 <= s.start_dt.hour < 23
            assert s.start_dt.date() == DAY
            assert s.start + s.estimated_seconds() <= dt.datetime(2026, 9, 7, 23).timestamp()
            assert 5 <= s.swipes <= 80
        slots = sorted(plan.slots, key=lambda s: s.start)
        for a, b in zip(slots, slots[1:], strict=False):
            assert b.start >= a.start + a.estimated_seconds() + 45 * 60
        counts.add(len(plan.slots))
        starts.update(s.start_dt.hour for s in plan.slots)
    assert len(counts) >= 3           # different number of sessions on different days
    assert len(starts) >= 6           # spread over the day, not always 09:xx


def test_plan_day_prefers_weighted_hours():
    weights = [0.0] * 24
    weights[20] = 1.0
    weights[21] = 1.0
    cfg = _cfg(p_rest_day=0.0, hour_weights=weights, ramp_days=0)
    hours = [s.start_dt.hour for seed in range(30) for s in plan_day(cfg, DAY, random.Random(seed)).slots]
    assert hours and set(hours) <= {20, 21}


def test_rest_days_and_ramp():
    cfg = _cfg(p_rest_day=1.0)
    plan = plan_day(cfg, DAY, random.Random(0))
    assert plan.rest and plan.slots == [] and plan.describe() == ["2026-09-07: rest day"]
    cfg = _cfg(p_rest_day=0.0, ramp_days=0)
    small = plan_day(cfg, DAY, random.Random(3), ramp=0.4)
    full = plan_day(cfg, DAY, random.Random(3), ramp=1.0)
    assert small.budget < full.budget


def test_ramp_factor_grows_from_first_auto_decision(storage):
    cfg = _cfg(ramp_days=10, ramp_start=0.5)
    assert ramp_factor(cfg, storage) == 0.5
    storage.upsert_profile(ProfileRecord(id="a"))
    storage.add_decision("a", "like", 0.9, "auto")
    first = storage.first_decision_ts("auto")
    assert ramp_factor(cfg, storage, now=first) == pytest.approx(0.5)
    assert ramp_factor(cfg, storage, now=first + 5 * 86400) == pytest.approx(0.75)
    assert ramp_factor(cfg, storage, now=first + 30 * 86400) == 1.0
    assert ramp_factor(_cfg(ramp_days=0), storage) == 1.0


def test_plan_roundtrip():
    plan = plan_day(_cfg(p_rest_day=0.0), DAY, random.Random(1))
    again = DayPlan.from_dict(plan.to_dict())
    assert again == plan


@dataclass
class FakeRunner:
    outcomes: list
    calls: list = None

    def __post_init__(self):
        self.calls = []

    def run_session(self, swipes):
        self.calls.append(swipes)
        reason = self.outcomes.pop(0) if self.outcomes else "planned"
        if isinstance(reason, tuple):
            reason, done = reason
        else:
            done = swipes
        return SessionResult(planned=swipes, done=done, reason=reason, error="x" if reason == "error" else "",
                             stats=RunStats())


class Clock:
    def __init__(self, start: dt.datetime):
        self.t = start.timestamp()
        self.slept = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept += s
        self.t += s


def _sched(storage, runner, clock, **schedule):
    cfg = _cfg(p_rest_day=0.0, ramp_days=0, **schedule)
    return Scheduler(cfg, storage, lambda: runner, rng=random.Random(5), now=clock.now, sleep=clock.sleep,
                     log=lambda m: None), cfg


def test_scheduler_runs_every_slot_of_the_day_and_persists_the_plan(storage):
    clock = Clock(dt.datetime(2026, 9, 7, 0, 5))
    runner = FakeRunner([])
    sched, _ = _sched(storage, runner, clock)
    plan = sched.plan_for(sched.today())
    n = len(plan.slots)
    assert n >= 1
    # a second scheduler instance (restart) sees the same plan
    other = Scheduler(sched.cfg, storage, lambda: runner, rng=random.Random(99), now=clock.now, sleep=clock.sleep,
                      log=lambda m: None)
    assert other.plan_for(sched.today()) == plan

    what = sched.run(max_days=1)
    assert what == "day_done"
    assert runner.calls == [s.swipes for s in sorted(plan.slots, key=lambda s: s.start)]
    saved = sched.load_plan(dt.date(2026, 9, 7))
    assert all(s.status == SLOT_DONE and s.done == s.swipes for s in saved.slots)
    # the browser was never open while waiting: sessions start at (or just after) their slot time
    assert clock.t > dt.datetime(2026, 9, 8).timestamp()


def test_scheduler_skips_slots_missed_while_asleep(storage):
    clock = Clock(dt.datetime(2026, 9, 7, 0, 5))
    runner = FakeRunner([])
    sched, cfg = _sched(storage, runner, clock, slot_grace_minutes=30)
    plan = sched.plan_for(sched.today())
    first = plan.next_pending()
    clock.t = first.start + 31 * 60          # laptop woke up too late
    what, delay = sched.tick()
    assert what == "missed" and delay == 0
    assert sched.load_plan(sched.today()).slots[[s.start for s in plan.slots].index(first.start)].status == SLOT_MISSED
    assert runner.calls == []


def test_out_of_likes_cancels_rest_of_day(storage):
    clock = Clock(dt.datetime(2026, 9, 7, 0, 5))
    runner = FakeRunner([("out_of_likes", 3)])
    sched, _ = _sched(storage, runner, clock)
    # force a plan with several slots
    plan = sched.plan_for(sched.today())
    while len(plan.slots) < 2:
        storage.delete_meta(f"dayplan:{plan.date}")
        sched.rng = random.Random(len(runner.calls) + sched.rng.randint(0, 10**6))
        plan = sched.plan_for(sched.today())
    sched.run(max_days=1)
    assert runner.calls == [sorted(plan.slots, key=lambda s: s.start)[0].swipes]
    saved = sched.load_plan(dt.date(2026, 9, 7))
    statuses = [s.status for s in sorted(saved.slots, key=lambda s: s.start)]
    assert statuses[0] == SLOT_DONE and set(statuses[1:]) == {SLOT_CANCELLED}
    assert saved.slots[0].done == 3 or sorted(saved.slots, key=lambda s: s.start)[0].done == 3


def test_account_notice_halts_until_resume(storage):
    clock = Clock(dt.datetime(2026, 9, 7, 0, 5))
    runner = FakeRunner(["account_notice"])
    sched, _ = _sched(storage, runner, clock)
    assert sched.run(max_days=3) == "halted"
    assert sched.halted()["reason"] == "account_notice"
    assert len(runner.calls) == 1
    # nothing runs while halted, even on later days
    clock.t += 3 * 86400
    assert sched.tick() == ("halted", 0.0)
    sched.resume()
    assert sched.halted() is None


def test_unsolved_challenge_pauses_then_halts(storage):
    clock = Clock(dt.datetime(2026, 9, 7, 0, 5))
    runner = FakeRunner(["captcha_unsolved", "captcha_unsolved"])
    sched, cfg = _sched(storage, runner, clock, max_unsolved_challenges=2, unsolved_challenge_pause_hours=(12, 12))
    sched.run(max_sessions=1)
    assert sched.halted() is None
    paused_until = sched.pause_until()
    assert paused_until == pytest.approx(clock.t + 12 * 3600)
    assert all(s.status == SLOT_CANCELLED for s in sched.load_plan(sched.today()).slots if s.status != SLOT_DONE)
    # next day: slots that fall inside the pause are skipped, the first one after it runs and halts
    sched.run(max_sessions=2)
    assert len(runner.calls) == 2
    assert sched.halted()["reason"] == "unsolved_challenges"


def test_solved_challenge_resets_streak_and_cools_down(storage):
    clock = Clock(dt.datetime(2026, 9, 7, 0, 5))
    runner = FakeRunner(["captcha_unsolved", "captcha_solved", "planned"])
    sched, cfg = _sched(storage, runner, clock, max_unsolved_challenges=2)
    sched.run(max_sessions=2)
    assert storage.get_meta("unsolved_challenge_streak") == 0
    assert sched.pause_until() > clock.t
    assert sched.halted() is None


def test_repeated_errors_halt(storage):
    clock = Clock(dt.datetime(2026, 9, 7, 0, 5))
    runner = FakeRunner(["error", "planned", "error", "error", "error"])
    sched, cfg = _sched(storage, runner, clock, max_consecutive_errors=3)
    sched.run(max_days=30)
    assert sched.halted()["reason"] == "errors"
    assert len(runner.calls) == 5  # the successful session reset the streak
