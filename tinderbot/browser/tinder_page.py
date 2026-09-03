"""Tinder web page driver: read the current card, like/nope with humanised input, dismiss popups.

Selectors are kept in one place because Tinder's DOM changes a few times a year. Every action tries
a list of candidates in order (accessibility attributes first: they are the most stable), and the
like/nope actions fall back to Tinder's own keyboard shortcuts.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field

from .humanize import HumanMouse, lognormal_delay
from .recs import RECS_URL_PATTERN, RecsQueue

RECS_URL = "https://tinder.com/app/recs"

SELECTORS = {
    "like": [
        'button[aria-label="Like"]',
        'button:has(span.Hidden:text-is("Like"))',
        'button:has-text("Like"):not(:has-text("Super"))',
    ],
    "nope": [
        'button[aria-label="Nope"]',
        'button:has(span.Hidden:text-is("Nope"))',
        'button:has-text("Nope")',
    ],
    "open_profile": ['button[aria-label*="Open Profile" i]', 'button[aria-label*="Show more" i]'],
    "close_profile": ['button[aria-label*="Close Profile" i]', 'button[aria-label*="Close" i]'],
    "name": ['[itemprop="name"]', 'h1 span:first-child'],
    "age": ['[itemprop="age"]', 'h1 span:nth-child(2)'],
    "login_button": ['a[href*="login"]', 'button:has-text("Log in")', 'div[aria-label="Log in"]'],
}

# Buttons that dismiss upsells / prompts. Text is matched case-insensitively.
POPUP_DISMISS_TEXTS = [
    "No Thanks", "Not interested", "Maybe later", "Not now", "Back to Tinder", "Keep swiping",
    "Keep Swiping", "Skip", "Dismiss", "Got it", "Remind me later", "I'll pass",
]
POPUP_DISMISS_SELECTORS = [
    'button[title="Back to Tinder"]',
    'div[role="dialog"] button[aria-label="Close" i]',
    'button[aria-label="Close" i]',
]
OUT_OF_LIKES_TEXTS = ["out of likes", "you're out of likes", "get more likes", "unlimited likes"]

# JS that finds the visible card's photos (inline background-image urls from Tinder's image CDN).
_JS_CARD_PHOTOS = """
() => {
  const urls = [];
  const seen = new Set();
  const inView = (el) => { const r = el.getBoundingClientRect();
        return r.width > 80 && r.height > 80 && r.bottom > 0 && r.right > 0 &&
               r.top < innerHeight && r.left < innerWidth; };
  document.querySelectorAll('[style*="background-image"]').forEach(el => {
    const m = /url\\(["']?([^"')]+)["']?\\)/.exec(el.style.backgroundImage || '');
    if (!m) return;
    const u = m[1];
    if (!/gotinder\\.com|tinder/.test(u)) return;
    if (seen.has(u) || !inView(el)) return;
    seen.add(u); urls.push(u);
  });
  return urls;
}
"""

_JS_CARD_TEXT = """
(sel) => {
  const pick = (s) => Array.from(document.querySelectorAll(s)).filter(e => {
    const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.top >= 0 && r.top < innerHeight;
  });
  for (const s of sel.name) { const els = pick(s); if (els.length) {
      let age = null;
      for (const a of sel.age) { const ae = pick(a); if (ae.length) { age = ae[0].innerText; break; } }
      return {name: els[0].innerText, age: age}; } }
  return null;
}
"""


@dataclass
class CardInfo:
    name: str = ""
    age: int | None = None
    photo_urls: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.name}|{self.age}|{self.photo_urls[0] if self.photo_urls else ''}"


class TinderPage:
    def __init__(self, page, rng: random.Random | None = None):
        self.page = page
        self.rng = rng or random.Random()
        self.mouse = HumanMouse(page, self.rng)
        self.queue = RecsQueue()
        self.human_actions: list[tuple[str, str]] = []  # ('like'|'pass', profile_id) seen on the network
        self._install_listeners()

    # ---- network observation --------------------------------------------------------
    def _install_listeners(self) -> None:
        def on_response(resp):
            try:
                if RECS_URL_PATTERN.search(resp.url) and resp.status == 200:
                    self.queue.add_payload(resp.text())
            except Exception:
                pass

        def on_request(req):
            m = re.search(r"api\.gotinder\.com/(like|pass)/([A-Za-z0-9]+)", req.url)
            if m:
                self.human_actions.append((m.group(1), m.group(2)))

        self.page.on("response", on_response)
        self.page.on("request", on_request)

    # ---- navigation / state -----------------------------------------------------------
    def goto_recs(self) -> None:
        if not self.page.url.startswith(RECS_URL):
            self.page.goto(RECS_URL, wait_until="domcontentloaded")

    def _first(self, key: str):
        for sel in SELECTORS[key]:
            loc = self.page.locator(sel).first
            try:
                if loc.count() and loc.is_visible():
                    return loc
            except Exception:
                continue
        return None

    def is_logged_in(self) -> bool:
        if "/app/" in self.page.url and self._first("like") is not None:
            return True
        return False

    def wait_for_login(self, max_minutes: float = 15.0, poll: float = 3.0) -> bool:
        deadline = time.time() + max_minutes * 60
        while time.time() < deadline:
            if self.is_logged_in():
                return True
            time.sleep(poll)
        return False

    def wait_ready(self, timeout_s: float = 30.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self.dismiss_popups()
            if self._first("like") is not None and self.current_card().name:
                return True
            time.sleep(0.8)
        return False

    # ---- reading the card ----------------------------------------------------------------
    def current_card(self) -> CardInfo:
        info = CardInfo()
        try:
            t = self.page.evaluate(_JS_CARD_TEXT, {"name": SELECTORS["name"], "age": SELECTORS["age"]})
            if t:
                info.name = (t.get("name") or "").strip()
                m = re.search(r"\d{2}", t.get("age") or "")
                info.age = int(m.group()) if m else None
            info.photo_urls = self.page.evaluate(_JS_CARD_PHOTOS) or []
        except Exception:
            pass
        return info

    def fetch_photo(self, url: str, timeout_ms: int = 15000) -> bytes | None:
        """Download through the browser's own request context (same cookies/headers as the app)."""
        try:
            r = self.page.context.request.get(url, timeout=timeout_ms)
            if r.ok:
                return r.body()
        except Exception:
            return None
        return None

    def screenshot_card(self) -> bytes | None:
        try:
            return self.page.screenshot(type="jpeg", quality=85)
        except Exception:
            return None

    # ---- human-like browsing ------------------------------------------------------------
    def browse_photos(self, n: int) -> int:
        """Advance through photos with Space (Tinder's shortcut) or a click on the card's right half."""
        done = 0
        for _ in range(max(0, n)):
            try:
                if self.rng.random() < 0.5:
                    self.page.keyboard.press("Space")
                else:
                    vp = self.page.viewport_size or {"width": 1280, "height": 800}
                    self.mouse.move_to(vp["width"] * self.rng.uniform(0.55, 0.68), vp["height"] * self.rng.uniform(0.3, 0.55))
                    self.page.mouse.down()
                    time.sleep(lognormal_delay(0.04, 0.12, self.rng))
                    self.page.mouse.up()
                done += 1
                time.sleep(lognormal_delay(0.6, 2.2, self.rng))
            except Exception:
                break
        return done

    def peek_profile(self) -> None:
        """Open the profile details, scroll a bit, close it again."""
        try:
            btn = self._first("open_profile")
            if btn is not None:
                self.mouse.click(btn)
            else:
                self.page.keyboard.press("ArrowUp")
            time.sleep(lognormal_delay(1.0, 3.0, self.rng))
            self.mouse.scroll(self.rng.uniform(150, 500))
            time.sleep(lognormal_delay(0.8, 2.5, self.rng))
            close = self._first("close_profile")
            if close is not None:
                self.mouse.click(close)
            else:
                self.page.keyboard.press("ArrowDown")
            time.sleep(lognormal_delay(0.3, 0.9, self.rng))
        except Exception:
            pass

    # ---- actions --------------------------------------------------------------------------
    def _press(self, key: str, shortcut: str) -> bool:
        loc = self._first(key)
        if loc is not None:
            try:
                self.mouse.click(loc)
                return True
            except Exception:
                pass
        try:
            self.page.keyboard.press(shortcut)
            return True
        except Exception:
            return False

    def like(self) -> bool:
        return self._press("like", "ArrowRight")

    def nope(self) -> bool:
        return self._press("nope", "ArrowLeft")

    # ---- popups ---------------------------------------------------------------------------
    def dismiss_popups(self) -> list[str]:
        handled: list[str] = []
        for sel in POPUP_DISMISS_SELECTORS:
            try:
                loc = self.page.locator(sel).first
                if loc.count() and loc.is_visible():
                    self.mouse.click(loc)
                    handled.append(sel)
                    time.sleep(lognormal_delay(0.4, 1.0, self.rng))
            except Exception:
                continue
        for text in POPUP_DISMISS_TEXTS:
            try:
                loc = self.page.get_by_role("button", name=re.compile(rf"^\s*{re.escape(text)}\s*$", re.IGNORECASE)).first
                if loc.count() and loc.is_visible():
                    self.mouse.click(loc)
                    handled.append(text)
                    time.sleep(lognormal_delay(0.4, 1.0, self.rng))
            except Exception:
                continue
        return handled

    def out_of_likes(self) -> bool:
        try:
            text = (self.page.evaluate("() => document.body ? document.body.innerText.slice(0, 20000) : ''") or "").lower()
        except Exception:
            return False
        return any(t in text for t in OUT_OF_LIKES_TEXTS) and "dialog" in (self.page.content()[:200000].lower())

    def wait_for_new_card(self, previous_key: str, timeout_s: float = 12.0) -> CardInfo | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            card = self.current_card()
            if card.name and card.key != previous_key:
                return card
            time.sleep(0.4)
        return None
