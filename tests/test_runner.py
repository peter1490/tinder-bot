import random

from tinderbot.browser.recs import RecsQueue
from tinderbot.browser.tinder_page import CardInfo
from tinderbot.config import CrushConfig
from tinderbot.crush import CrushPolicy
from tinderbot.likeness.scorer import Verdict
from tinderbot.runner import Runner, RunStats
from tinderbot.storage import ProfileRecord


class FakeStorage:
    def __init__(self):
        self.decisions = []
        self.events = []

    def add_decision(self, *args, **kwargs):
        self.decisions.append((args, kwargs))

    def log_event(self, kind, detail):
        self.events.append((kind, detail))


class FakePacer:
    def plan_profile(self, photo_count, bio_len, like=None):
        return {"browse_photos": 0, "open_profile": False, "read_seconds": 0, "micro_break": 0}

    def post_action_delay(self):
        return 0

    def should_end_early(self, done, planned):
        return False


class FakeGovernor:
    def __init__(self, allow=True):
        self.allow = allow
        self.downgraded = 0

    def allow_like(self):
        if not self.allow:
            self.downgraded += 1
        return self.allow


class FakeScorer:
    def __init__(self):
        self.retrains = 0

    def maybe_retrain(self):
        self.retrains += 1


class FakeMouse:
    def wiggle(self):
        raise AssertionError("wiggle should not be selected by the deterministic RNG")


class StuckPage:
    def __init__(self):
        self.card = CardInfo("Ana", 25, ["https://example.test/first.jpg"])
        self.mouse = FakeMouse()
        self.queue = RecsQueue()
        self.waited_for = None
        self.clicks = 0
        self.super_likes_remaining = None

    def dismiss_popups(self):
        return []

    def out_of_likes(self):
        return False

    def current_card(self):
        return self.card

    def browse_photos(self, n):
        return 0

    def peek_profile(self):
        return None

    def like(self):
        self.clicks += 1
        return True

    def nope(self):
        self.clicks += 1
        return True

    def wait_for_new_card(self, previous_key, timeout_s):
        self.waited_for = previous_key
        return None

    def reload_recs(self, timeout_s):
        return False


def test_unconfirmed_swipe_is_not_counted_or_persisted(monkeypatch):
    runner = Runner.__new__(Runner)
    runner.stats = RunStats()
    runner.rng = random.Random(0)  # first random() is > 0.5, so no mouse wiggle
    runner.storage = FakeStorage()
    runner.pacer = FakePacer()
    runner.governor = FakeGovernor()
    runner.scorer = FakeScorer()
    runner.crush = CrushPolicy(CrushConfig(), runner.storage)
    runner._check_challenge = lambda tp, unattended=False: "ok"
    runner.resolve_profile = lambda tp, card: ProfileRecord(id="ana", name="Ana", age=25)
    runner.score_profile = lambda tp, profile: (
        Verdict(like=True, score=0.8, prior=0.8, learned=None),
        {"score": 0.8},
    )
    monkeypatch.setattr("tinderbot.runner.time.sleep", lambda seconds: None)
    page = StuckPage()

    assert runner._swipe_session(page, 1) == 0
    assert page.clicks == 1
    assert page.waited_for == page.card.key
    assert runner.stats.total == 0
    assert runner.stats.skipped == 1
    assert runner.storage.decisions == []
    assert runner.storage.events[0][0] == "swipe_unconfirmed"
    assert runner.scorer.retrains == 0
    assert runner.stop_reason == "swipe_unconfirmed"


class AdvancingPage(StuckPage):
    """A page whose card advances after every click, so swipes get confirmed."""

    def __init__(self, names=("Ana", "Bea", "Cleo", "Dee")):
        super().__init__()
        self.names = list(names)
        self.i = 0
        self.card = CardInfo(self.names[0], 25, [])
        self.actions = []

    def like(self):
        self.actions.append("like")
        return True

    def nope(self):
        self.actions.append("nope")
        return True

    def wait_for_new_card(self, previous_key, timeout_s):
        self.i += 1
        if self.i >= len(self.names):
            return None
        self.card = CardInfo(self.names[self.i], 25, [])
        return self.card


def _bare_runner(monkeypatch, governor):
    runner = Runner.__new__(Runner)
    runner.stats = RunStats()
    runner.rng = random.Random(0)
    runner.storage = FakeStorage()
    runner.pacer = FakePacer()
    runner.governor = governor
    runner.scorer = FakeScorer()
    runner.crush = CrushPolicy(CrushConfig(), runner.storage)
    runner._check_challenge = lambda tp, unattended=False: "ok"
    runner.resolve_profile = lambda tp, card: ProfileRecord(id=card.name.lower(), name=card.name, age=25)
    runner.score_profile = lambda tp, profile: (Verdict(like=True, score=0.8, prior=0.8, learned=None), {"s": 0.8})
    monkeypatch.setattr("tinderbot.runner.time.sleep", lambda seconds: None)
    monkeypatch.setattr("tinderbot.runner.random.Random.random", lambda self: 0.9)  # no wiggle
    return runner


def test_like_ratio_governor_turns_likes_into_nopes_but_keeps_verdict_label(monkeypatch):
    runner = _bare_runner(monkeypatch, FakeGovernor(allow=False))
    page = AdvancingPage()
    assert runner._swipe_session(page, 3) == 3
    assert page.actions == ["nope", "nope", "nope"]
    assert runner.stats.noped == 3 and runner.stats.liked == 0
    for args, kwargs in runner.storage.decisions:
        assert args[1] == "nope"                # what was done
        assert kwargs["label"] == 1             # what the scorer thought
        assert "like_ratio_cap" in args[4]
    assert runner.stop_reason == "planned"


def test_governor_lets_likes_through_when_allowed(monkeypatch):
    runner = _bare_runner(monkeypatch, FakeGovernor(allow=True))
    page = AdvancingPage()
    assert runner._swipe_session(page, 2) == 2
    assert page.actions == ["like", "like"]
    assert runner.stats.liked == 2


class FakeSession:
    closed = False

    def __init__(self):
        self.page = object()
        self.backend = "fake"

    def close(self):
        FakeSession.closed = True


def test_run_session_reports_reason_and_closes_browser(monkeypatch, cfg, storage):
    from tinderbot.browser.pacing import Pacer

    runner = Runner.__new__(Runner)
    runner.cfg = cfg
    runner.storage = storage
    runner.rng = random.Random(1)
    runner.pacer = Pacer(cfg.pacing, storage, runner.rng)
    runner.governor = FakeGovernor()
    runner.extractor = type("E", (), {"reference_summary": lambda self: "none"})()
    FakeSession.closed = False
    monkeypatch.setattr("tinderbot.runner.launch", lambda cfg, url: FakeSession())
    monkeypatch.setattr("tinderbot.runner.TinderPage", lambda page, rng, keyboard_pref=0.0: object())
    monkeypatch.setattr("tinderbot.runner.time.sleep", lambda seconds: None)
    runner._ensure_logged_in = lambda tp, wait_minutes=0.0: True
    runner._warm_up = lambda tp: None

    def fake_swipe(tp, n, unattended=False):
        runner.stop_reason = "out_of_likes"
        return 4

    runner._swipe_session = fake_swipe
    res = runner.run_session(20)
    assert res.done == 4 and res.reason == "out_of_likes" and res.planned == 20
    assert FakeSession.closed is True
    assert storage.count_events("session_start", 0) == 1 and storage.count_events("session_end", 0) == 1

    # an exception inside the session is reported, never raised
    def boom(tp, n, unattended=False):
        raise RuntimeError("browser died")

    runner._swipe_session = boom
    FakeSession.closed = False
    res = runner.run_session(10)
    assert res.reason == "error" and "browser died" in res.error
    assert FakeSession.closed is True
    assert storage.count_events("session_error", 0) == 1

    # lost login halts (reported through the reason)
    runner._ensure_logged_in = lambda tp, wait_minutes=0.0: False
    res = runner.run_session(10)
    assert res.reason == "needs_login" and res.halting


class QuietMouse:
    def wiggle(self):
        pass


class CrushPage(StuckPage):
    """Advances on every action; Super Like either works (recording the note) or shows the upsell."""

    def __init__(self, mode="ok"):
        super().__init__()
        self.mouse = QuietMouse()
        self.mode = mode
        self.super_likes = []
        self.n = 0

    def current_card(self):
        return CardInfo(f"P{self.n}", 25, [f"https://example.test/{self.n}.jpg"])

    def super_like(self, note=None):
        if self.mode == "unavailable":
            return "unavailable"
        self.super_likes.append(note)
        return "sent_note" if note else "sent"

    def wait_for_new_card(self, previous_key, timeout_s):
        self.n += 1
        return self.current_card()


def _crush_runner(storage, page, scores, monkeypatch):
    runner = Runner.__new__(Runner)
    runner.stats = RunStats()
    runner.rng = random.Random(0)
    runner.storage = storage
    runner.pacer = FakePacer()
    runner.governor = FakeGovernor()
    runner.scorer = FakeScorer()
    runner.crush = CrushPolicy(CrushConfig(messages=["Hey {name}, hi!"]), storage, random.Random(0))
    runner._check_challenge = lambda tp, unattended=False: "ok"

    def resolve(tp, card):
        p = ProfileRecord(id=card.name.lower(), name=card.name, age=25)
        storage.upsert_profile(p)
        return p

    def score(tp, profile):
        s = scores[int(profile.id[1:])]
        return Verdict(like=s >= 0.55, score=s, prior=s, learned=s, crush=s >= 0.9, super_crush=s >= 0.95), {"s": s}

    runner.resolve_profile = resolve
    runner.score_profile = score
    monkeypatch.setattr("tinderbot.runner.time.sleep", lambda seconds: None)
    return runner


def test_one_super_like_and_one_note_per_day(storage, monkeypatch):
    page = CrushPage()
    runner = _crush_runner(storage, page, [0.97, 0.99, 0.92, 0.3], monkeypatch)
    assert runner._swipe_session(page, 4) == 4
    # first super crush: Super Like + note; the next crushes of the day are plain likes
    assert page.super_likes == ["Hey P0, hi!"]
    actions = [storage.decisions_for_profile(f"p{i}")[0]["action"] for i in range(4)]
    assert actions == ["superlike", "like", "like", "nope"]
    assert storage.count_events("superlike", 0) == 1 and storage.count_events("crush_message", 0) == 1
    assert (runner.stats.liked, runner.stats.super_liked, runner.stats.notes, runner.stats.noped) == (3, 1, 1, 1)
    assert runner.crush.plan(Verdict(like=True, score=0.99, prior=0.99, learned=0.99, crush=True, super_crush=True)) is None


def test_crush_without_note_budget_still_gets_the_super_like(storage, monkeypatch):
    page = CrushPage()
    storage.log_event("crush_message", {"profile_id": "earlier"})     # today's note already used
    runner = _crush_runner(storage, page, [0.99, 0.91], monkeypatch)
    assert runner._swipe_session(page, 2) == 2
    assert page.super_likes == [None]
    assert storage.decisions_for_profile("p0")[0]["action"] == "superlike"
    assert storage.decisions_for_profile("p1")[0]["action"] == "like"
    assert storage.count_events("crush_message", 0) == 1


def test_super_like_unavailable_falls_back_to_like_for_the_day(storage, monkeypatch):
    page = CrushPage(mode="unavailable")
    runner = _crush_runner(storage, page, [0.99, 0.99], monkeypatch)
    assert runner._swipe_session(page, 2) == 2
    assert page.clicks == 2 and page.super_likes == []
    assert [storage.decisions_for_profile(f"p{i}")[0]["action"] for i in range(2)] == ["like", "like"]
    assert storage.count_events("superlike", 0) == 0
    assert runner.crush.unavailable_today() and not runner.crush.can_super_like()


def test_tinder_reported_balance_blocks_the_attempt(storage, monkeypatch):
    page = CrushPage()
    page.super_likes_remaining = 0
    runner = _crush_runner(storage, page, [0.99], monkeypatch)
    assert runner._swipe_session(page, 1) == 1
    assert page.super_likes == [] and storage.decisions_for_profile("p0")[0]["action"] == "like"


def test_note_templates():
    policy = CrushPolicy(CrushConfig(messages=["Hi {name}! Coffee?"]), None, random.Random(0))
    assert policy.note_for(ProfileRecord(id="x", name="Ana Maria")) == "Hi Ana! Coffee?"
    assert policy.note_for(ProfileRecord(id="x", name="")) == "Hi! Coffee?"
