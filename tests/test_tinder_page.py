"""Drive the page layer against a local mock of Tinder's recs screen (real Chromium, headless)."""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from tests.conftest import CHROME
from tinderbot.browser.captcha import CaptchaPolicy, Challenge, detect_challenge
from tinderbot.browser.tinder_page import POPUP_DISMISS_TEXTS, CardInfo, TinderPage
from tinderbot.config import CaptchaConfig

FIXTURE = Path(__file__).parent / "fixtures" / "mock_tinder.html"


def test_card_key_does_not_change_when_browsing_photos():
    first_photo = CardInfo(name="Ana", age=25, photo_urls=["https://example.test/first.jpg"])
    later_photo = CardInfo(name=" Ana ", age=25, photo_urls=["https://example.test/later.jpg"])
    next_profile = CardInfo(name="Bea", age=25, photo_urls=["https://example.test/first.jpg"])

    assert first_photo.key == later_photo.key
    assert first_photo.key != next_profile.key


def test_french_super_like_upsell_can_be_dismissed():
    assert "Non merci" in POPUP_DISMISS_TEXTS


@pytest.fixture(scope="module")
def page():
    from playwright.sync_api import sync_playwright

    if not os.path.exists(CHROME):
        pytest.skip("no chromium available")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=CHROME)
        pg = browser.new_page(viewport={"width": 1000, "height": 800})
        pg.goto(FIXTURE.as_uri())
        yield pg
        browser.close()


def test_read_card_and_swipe_with_mouse(page):
    tp = TinderPage(page, random.Random(0))
    tp.mouse._sleep = lambda s: None  # no real waiting in tests
    card = tp.current_card()
    assert card.name == "Ana" and card.age == 25
    assert card.photo_urls == ["https://images-ssl.gotinder.com/u/aaa/640x800.jpg?x=1"]
    assert tp.like()
    nxt = tp.wait_for_new_card(card.key, timeout_s=3)
    assert nxt and nxt.name == "Bea" and nxt.age == 31
    assert tp.nope()
    assert page.evaluate("window.actions") == ["like", "nope"]
    # the upsell dialog appeared after the second swipe -> dismissed by text
    handled = tp.dismiss_popups()
    assert "No Thanks" in handled
    assert page.evaluate("window.actions")[-1] == "dismiss"
    assert page.locator("#dialog").is_visible() is False
    assert tp.dismiss_popups() == []


def test_browse_photos_and_keyboard_fallback(page):
    tp = TinderPage(page, random.Random(1))
    tp.mouse._sleep = lambda s: None
    page.evaluate("window.actions = []")
    n = tp.browse_photos(2)
    assert n == 2
    # hide the buttons so the keyboard fallback is used
    page.evaluate("document.querySelector('.gamepad').style.display='none'")
    try:
        assert tp.like()
        assert page.evaluate("window.actions")[-1] == "like"
    finally:
        page.evaluate("document.querySelector('.gamepad').style.display='flex'")


def test_captcha_detection_and_policy(page, storage):
    assert detect_challenge(page) is None
    page.evaluate("window.showCaptcha()")
    ch = detect_challenge(page)
    assert ch and ch.kind == "captcha"
    policy = CaptchaPolicy(CaptchaConfig(wait_for_human_max_minutes=0.05, notify_desktop=False, sound=False), storage,
                           random.Random(0))
    ticks = {"n": 0}

    def fake_sleep(s):
        ticks["n"] += 1
        if ticks["n"] == 2:
            page.evaluate("window.hideCaptcha()")

    assert policy.handle(page, ch, sleep=fake_sleep) == "solved"
    assert policy.slowdown > 1.0
    assert storage.count_events("captcha", 0) == 1 and storage.count_events("captcha_solved", 0) == 1
    # account-level notice stops the run
    assert policy.handle(page, Challenge("account", "banned"), sleep=fake_sleep) == "stop"
    # timeout path
    page.evaluate("window.showCaptcha()")
    assert policy.handle(page, ch, sleep=lambda s: None) in ("timeout", "stop")
    page.evaluate("window.hideCaptcha()")


def test_out_of_likes_false_on_normal_card(page):
    tp = TinderPage(page)
    assert tp.out_of_likes() is False
