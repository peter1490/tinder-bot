import datetime as dt
import random

from tinderbot.browser.pacing import Pacer
from tinderbot.config import PacingConfig
from tinderbot.storage import ProfileRecord


def test_budgets_and_sessions(storage):
    cfg = PacingConfig(swipes_per_session=(5, 5), sessions_per_day=2, max_swipes_per_day=8)
    p = Pacer(cfg, storage, random.Random(0))
    plan = p.start_session()
    assert plan.swipes == 5
    for i in range(5):
        storage.upsert_profile(ProfileRecord(id=f"a{i}"))
        storage.add_decision(f"a{i}", "like", 0.9, "auto")
    p.end_session(5)
    assert p.swipes_today() == 5 and p.remaining_today() == 3
    plan2 = p.start_session()
    assert plan2.swipes == 3  # capped by the daily budget
    p.end_session(3)
    assert p.start_session() is None  # sessions_per_day reached


def test_max_override_and_active_hours(storage):
    cfg = PacingConfig(swipes_per_session=(50, 50), active_hours=(9, 18))
    p = Pacer(cfg, storage, random.Random(0))
    assert p.start_session(max_override=7).swipes == 7
    assert p.in_active_hours(dt.datetime(2026, 1, 1, 12))
    assert not p.in_active_hours(dt.datetime(2026, 1, 1, 20))
    secs = p.seconds_until_active(dt.datetime(2026, 1, 1, 20))
    assert 12 * 3600 <= secs <= 14 * 3600
    assert p.seconds_until_active(dt.datetime(2026, 1, 1, 12)) == 0


def test_profile_plan_and_slowdown(storage):
    cfg = PacingConfig(p_browse_photos=1.0, max_photos_browsed=3, p_open_profile=0.0, p_micro_break=0.0,
                       base_seconds=(1, 1), per_photo_seconds=(1, 1), per_bio_char_seconds=0.0)
    p = Pacer(cfg, storage, random.Random(0))
    plan = p.plan_profile(photo_count=5, bio_len=0)
    assert 1 <= plan["browse_photos"] <= 3
    assert plan["read_seconds"] == 1 + plan["browse_photos"]
    assert plan["micro_break"] == 0.0 and plan["open_profile"] is False
    p.slowdown = 2.0
    plan = p.plan_profile(photo_count=1, bio_len=0)
    assert plan["browse_photos"] == 0 and plan["read_seconds"] == 2.0
