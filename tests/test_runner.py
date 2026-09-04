import random

from tinderbot.browser.recs import RecsQueue
from tinderbot.browser.tinder_page import CardInfo
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
    def plan_profile(self, photo_count, bio_len):
        return {"browse_photos": 0, "open_profile": False, "read_seconds": 0, "micro_break": 0}

    def post_action_delay(self):
        return 0


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
    runner.scorer = FakeScorer()
    runner._check_challenge = lambda tp: "ok"
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
