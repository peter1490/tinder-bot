import datetime as dt
import random

from tinderbot.browser.pacing import LikeGovernor, Pacer
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


def test_sessions_per_day_accepts_int_and_range():
    assert PacingConfig(sessions_per_day=3).sessions_per_day == (3, 3)
    assert PacingConfig(sessions_per_day=[1, 4]).sessions_per_day == (1, 4)


def test_budget_counts_shadow_swipes_too(storage):
    cfg = PacingConfig(max_swipes_per_day=10)
    p = Pacer(cfg, storage, random.Random(0))
    storage.upsert_profile(ProfileRecord(id="m1"))
    storage.add_decision("m1", "like", 0.9, "manual")
    storage.upsert_profile(ProfileRecord(id="a1"))
    storage.add_decision("a1", "nope", 0.1, "auto")
    assert p.swipes_today() == 2 and p.remaining_today() == 8


def test_verdict_aware_dwell_and_persona(storage):
    cfg = PacingConfig(p_browse_photos=0.0, p_open_profile=0.0, p_micro_break=0.0,
                       base_seconds=(2, 2), per_bio_char_seconds=0.0,
                       like_dwell_multiplier=(1.5, 1.5), nope_dwell_multiplier=(0.5, 0.5),
                       session_tempo=(1.2, 1.2), keyboard_preference=(0.3, 0.3))
    p = Pacer(cfg, storage, random.Random(0))
    assert p.plan_profile(1, 0)["read_seconds"] == 2.0
    assert p.plan_profile(1, 0, like=True)["read_seconds"] == 3.0
    assert p.plan_profile(1, 0, like=False)["read_seconds"] == 1.0
    plan = p.begin_session(20)
    assert plan.swipes == 20
    assert p.persona.tempo == 1.2 and p.persona.keyboard_pref == 0.3
    assert p.plan_profile(1, 0)["read_seconds"] == 2.4
    assert storage.count_events("session_start", 0) == 1


def test_begin_session_respects_budget(storage):
    cfg = PacingConfig(max_swipes_per_day=3)
    p = Pacer(cfg, storage, random.Random(0))
    assert p.begin_session(50).swipes == 3
    for i in range(3):
        storage.upsert_profile(ProfileRecord(id=f"x{i}"))
        storage.add_decision(f"x{i}", "nope", 0.1, "auto")
    assert p.begin_session(50) is None


def test_session_swipes_are_right_skewed(storage):
    cfg = PacingConfig(swipes_per_session=(15, 80))
    p = Pacer(cfg, storage, random.Random(0))
    draws = [p.draw_session_swipes() for _ in range(400)]
    assert all(15 <= d <= 80 for d in draws)
    median = sorted(draws)[200]
    assert median < (15 + 80) / 2 + 5  # bulk of sessions on the short side


def test_like_governor_caps_recent_like_share(storage):
    cfg = PacingConfig(max_like_ratio=0.5, like_ratio_window=10, like_ratio_min_samples=4)
    g = LikeGovernor(cfg, storage)
    assert g.allow_like()  # not enough samples yet
    for i in range(6):
        storage.upsert_profile(ProfileRecord(id=f"p{i}"))
        storage.add_decision(f"p{i}", "like" if i < 4 else "nope", 0.9, "auto")
    # 4 likes / 6 -> another like would be 5/7 > 0.5
    assert not g.allow_like()
    assert g.downgraded == 1
    for i in range(6, 10):
        storage.upsert_profile(ProfileRecord(id=f"p{i}"))
        storage.add_decision(f"p{i}", "nope", 0.1, "auto")
    # 4 likes / 10 -> a like makes 5/10 (oldest 'like' drops out of the full window: 4/10) -> ok
    assert g.allow_like()
    assert LikeGovernor(PacingConfig(max_like_ratio=1.0), storage).allow_like()


def test_should_end_early_only_after_half(storage):
    cfg = PacingConfig(p_end_session_early=1.0)
    p = Pacer(cfg, storage, random.Random(0))
    assert not p.should_end_early(4, 10)
    assert p.should_end_early(5, 10)
    assert not Pacer(PacingConfig(p_end_session_early=0.0), storage).should_end_early(9, 10)
