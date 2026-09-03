"""Captcha / account-challenge handling: detect, hand over to the human, back off.

Tinder's challenge is Arkose Labs (FunCaptcha). It is risk-scored: it appears when the session looks
automated (input telemetry, request cadence, fresh browser profile, device inconsistencies). The bot
therefore does NOT try to solve it. It stops, tells you, waits for you to solve it in the same
window, records the event and slows down for the rest of the day so it stops being triggered.
"""

from __future__ import annotations

import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..config import CaptchaConfig
from ..storage import Storage

CAPTCHA_FRAME_HINTS = ("arkoselabs", "funcaptcha", "hcaptcha", "recaptcha", "captcha-delivery", "challenge")
CAPTCHA_TEXT_HINTS = (
    "verify you're human", "verify you are human", "let's make sure you're a real person",
    "make sure you're a real person", "are you a human", "solve this puzzle", "complete the puzzle",
    "press and hold", "security check", "human verification",
)
ACCOUNT_TEXT_HINTS = (
    "your account has been banned", "account under review", "we've noticed unusual activity",
    "you've been logged out", "something went wrong. please try again later",
)


@dataclass
class Challenge:
    kind: str          # 'captcha' | 'account'
    detail: str


def detect_challenge(page) -> Challenge | None:
    """Look for a challenge iframe or challenge wording on the current page (cheap, no navigation)."""
    try:
        for frame in page.frames:
            url = (frame.url or "").lower()
            if any(h in url for h in CAPTCHA_FRAME_HINTS) and "tinder.com" not in url:
                return Challenge("captcha", f"iframe {url[:80]}")
        text = page.evaluate("() => (document.body && document.body.innerText || '').slice(0, 20000)") or ""
        low = text.lower()
        for h in CAPTCHA_TEXT_HINTS:
            if h in low:
                return Challenge("captcha", f"text: {h}")
        for h in ACCOUNT_TEXT_HINTS:
            if h in low:
                return Challenge("account", f"text: {h}")
    except Exception as e:  # page navigating / closed
        return Challenge("unknown", f"detect error: {e}") if "closed" in str(e).lower() else None
    return None


def notify(title: str, message: str, cfg: CaptchaConfig) -> None:
    print(f"\n\a=== {title}: {message} ===", file=sys.stderr, flush=True)
    if cfg.sound:
        for _ in range(3):
            sys.stderr.write("\a")
            sys.stderr.flush()
            time.sleep(0.25)
    if cfg.notify_desktop:
        try:
            from plyer import notification  # optional extra

            notification.notify(title=title, message=message, app_name="tinderbot", timeout=15)
        except Exception:
            pass


class CaptchaPolicy:
    """Daily challenge budget + cooldown/slow-down after each challenge."""

    def __init__(self, cfg: CaptchaConfig, storage: Storage, rng: random.Random | None = None):
        self.cfg = cfg
        self.storage = storage
        self.rng = rng or random.Random()
        self.slowdown = 1.0  # multiplier applied to all pauses for the rest of the day

    def challenges_today(self) -> int:
        midnight = time.time() - (time.time() % 86400)
        return self.storage.count_events("captcha", midnight)

    def handle(self, page, challenge: Challenge, sleep: Callable[[float], None] = time.sleep) -> str:
        """Block until the human solved the challenge (or timeout). Returns 'solved' | 'timeout' | 'stop'."""
        self.storage.log_event("captcha", {"kind": challenge.kind, "detail": challenge.detail})
        n = self.challenges_today()
        if challenge.kind == "account":
            notify("tinderbot stopped", f"Account notice on screen: {challenge.detail}", self.cfg)
            return "stop"
        notify("tinderbot needs you", "Tinder is showing a human-verification challenge. Solve it in the browser window.", self.cfg)
        try:
            page.bring_to_front()
        except Exception:
            pass
        deadline = time.time() + self.cfg.wait_for_human_max_minutes * 60
        while time.time() < deadline:
            sleep(3)
            if detect_challenge(page) is None:
                self.storage.log_event("captcha_solved", {"after_s": int(deadline - time.time())})
                self.slowdown = min(3.0, self.slowdown * 1.6)
                if n >= self.cfg.max_captchas_per_day:
                    notify("tinderbot stopping for today", f"{n} challenges today: reached max_captchas_per_day", self.cfg)
                    return "stop"
                return "solved"
        return "timeout"

    def cooldown_seconds(self) -> float:
        lo, hi = self.cfg.cooldown_after_captcha_minutes
        return self.rng.uniform(lo, hi) * 60
