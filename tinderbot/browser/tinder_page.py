"""Tinder web page driver: read the current card, like/nope with humanised input, dismiss popups.

Selectors are kept in one place because Tinder's DOM changes a few times a year. Every action tries
a list of candidates in order (accessibility attributes first: they are the most stable), and the
like/nope actions fall back to Tinder's own keyboard shortcuts.
"""

from __future__ import annotations

import json
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
    "super_like": [
        'button[aria-label="Super Like"]',
        'button:has(span.Hidden:text-is("Super Like"))',
        'button:has-text("Super Like")',
    ],
    # the "add a note to your Super Like" composer (Tinder shows it right after the Super Like click)
    "note_input": [
        'div[role="dialog"] textarea',
        'textarea[placeholder*="note" i]',
        'textarea[placeholder*="message" i]',
        'div[role="dialog"] [contenteditable="true"]',
    ],
    "open_profile": ['button[aria-label*="Open Profile" i]', 'button[aria-label*="Show more" i]'],
    "close_profile": ['button[aria-label*="Close Profile" i]', 'button[aria-label*="Close" i]'],
    "name": ['[itemprop="name"]', 'h1 span:first-child'],
    "age": ['[itemprop="age"]', 'h1 span:nth-child(2)'],
    "login_button": ['a[href*="login"]', 'button:has-text("Log in")', 'div[aria-label="Log in"]'],
}

# Buttons that send the Super Like from the note composer (with or without a note typed).
NOTE_SEND_TEXTS = ["Send Super Like", "Send note", "Send", "Super Like", "Envoyer le Super Like", "Envoyer"]
# Wording of the upsell Tinder shows instead of sending when no Super Like is available.
SUPERLIKE_UPSELL_TEXTS = [
    "out of super likes", "get super likes", "get more super likes", "buy super likes", "more super likes",
    "unlock super likes", "super likes to get noticed", "upgrade to send",
    "plus de super likes", "obtenir des super likes", "acheter des super likes", "obtenez des super likes",
]

# Buttons that dismiss upsells / prompts. Text is matched case-insensitively.
POPUP_DISMISS_TEXTS = [
    "No Thanks", "Not interested", "Maybe later", "Not now", "Back to Tinder", "Keep swiping",
    "Keep Swiping", "Skip", "Dismiss", "Got it", "Remind me later", "I'll pass",
    # Common French equivalents shown when Tinder follows the browser locale.
    "Non merci", "Pas maintenant", "Peut-être plus tard", "Retour à Tinder", "Continuer de swiper",
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
        # The visible photo URL changes when a profile's gallery is browsed.  It
        # therefore cannot be part of the DOM card identity: doing so can make
        # the current profile look like a new card before a swipe has advanced
        # Tinder.  Name + age is deliberately conservative; in the rare case
        # of consecutive identical values, confirmation times out instead of
        # risking a duplicate action.
        return f"{self.name.strip().casefold()}|{self.age}"


_JS_FETCH_PHOTO = """
async (url) => {
  const r = await fetch(url, {credentials: 'include', cache: 'force-cache'});
  if (!r.ok) return null;
  const b = await r.blob();
  return await new Promise((res) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(b); });
}
"""


class TinderPage:
    def __init__(self, page, rng: random.Random | None = None, keyboard_pref: float = 0.0):
        self.page = page
        self.rng = rng or random.Random()
        self.mouse = HumanMouse(page, self.rng)
        self.queue = RecsQueue()
        # Share of actions done with the keyboard shortcuts rather than the mouse (per-session persona).
        self.keyboard_pref = keyboard_pref
        self._page_fetch_failures = 0
        self.human_actions: list[tuple[str, str]] = []  # ('like'|'pass', profile_id) seen on the network
        self.human_actions: list[tuple[str, str]] = []  # ('like'|'superlike'|'pass', profile_id) seen on the network
        self.super_likes_remaining: int | None = None    # as last reported by Tinder's own responses
        self._sleep = time.sleep
        self._install_listeners()

    # ---- network observation --------------------------------------------------------
    def _install_listeners(self) -> None:
        def on_response(resp):
            try:
                url = resp.url
                if resp.status != 200:
                    return
                if RECS_URL_PATTERN.search(url):
                    self.queue.add_payload(resp.text())
                elif "api.gotinder.com" in url and ("/like/" in url or "/profile" in url):
                    self._observe_super_like_balance(resp.text())
            except Exception:
                pass

        def on_request(req):
            m = re.search(r"api\.gotinder\.com/(like|pass)/([A-Za-z0-9]+)(/super)?", req.url)
            if m:
                self.human_actions.append(("superlike" if m.group(3) else m.group(1), m.group(2)))

        self.page.on("response", on_response)
        self.page.on("request", on_request)

    def _observe_super_like_balance(self, body: str) -> None:
        """Like / profile responses carry ``super_likes.remaining``; remember it (never requested by us)."""
        if "super_likes" not in body:
            return
        try:
            data = json.loads(body)
        except ValueError:
            return
        if isinstance(data, dict) and isinstance(data.get("data"), dict):
            data = data["data"]
        sl = data.get("super_likes") if isinstance(data, dict) else None
        if isinstance(sl, dict) and isinstance(sl.get("remaining"), int):
            self.super_likes_remaining = int(sl["remaining"])

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

    def reload_recs(self, timeout_s: float = 30.0) -> bool:
        """Reload recommendations after a stalled transition and wait for a usable card."""
        try:
            self.page.reload(wait_until="domcontentloaded")
        except Exception:
            return False
        return self.wait_ready(timeout_s)

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
        """Download a photo the way the app itself would.

        First choice is ``fetch()`` inside the page: it reuses the browser's HTTP cache (the card's
        photos are usually already there) and sends exactly the headers the web app sends.  If the
        CDN refuses cross-origin reads the browser context's request API is used instead (same
        cookies, slightly different headers).
        """
        if self._page_fetch_failures < 3:
            try:
                data_url = self.page.evaluate(_JS_FETCH_PHOTO, url)
                if isinstance(data_url, str) and "," in data_url:
                    import base64

                    return base64.b64decode(data_url.split(",", 1)[1])
                self._page_fetch_failures += 1
            except Exception:
                self._page_fetch_failures += 1
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
        use_keys = self.keyboard_pref if self.keyboard_pref > 0 else 0.5
        for _ in range(max(0, n)):
            try:
                if self.rng.random() < use_keys:
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
        loc = None if self.rng.random() < self.keyboard_pref else self._first(key)
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

    def _dialog_text(self) -> str:
        try:
            return (self.page.evaluate(
                "() => Array.from(document.querySelectorAll('[role=dialog], [aria-modal=true]'))"
                ".filter(d => d.getBoundingClientRect().height > 0).map(d => d.innerText).join('\\n')"
            ) or "").lower()
        except Exception:
            return ""

    @staticmethod
    def is_super_like_upsell(text: str) -> bool:
        t = text.lower()
        if any(h in t for h in SUPERLIKE_UPSELL_TEXTS):
            return True
        # generic: a dialog about Super Likes that quotes a price is a shop, not a confirmation
        return "super like" in t and re.search(r"[$€£]\s?\d|\d\s?[$€£]", t) is not None

    def _button_by_text(self, texts: list[str]):
        """First *visible* button whose accessible name is one of ``texts`` (hidden duplicates are common)."""
        for text in texts:
            try:
                loc = self.page.get_by_role("button", name=re.compile(rf"^\s*{re.escape(text)}\s*$", re.IGNORECASE))
                for i in range(min(loc.count(), 8)):
                    if loc.nth(i).is_visible():
                        return loc.nth(i)
            except Exception:
                continue
        return None

    def type_like_human(self, text: str) -> None:
        for ch in text:
            self.page.keyboard.type(ch)
            self._sleep(lognormal_delay(0.04, 0.18, self.rng))

    def super_like(self, note: str | None = None) -> str:
        """Press Super Like, then handle whatever Tinder shows next.

        Returns ``'sent'`` (plain Super Like), ``'sent_note'`` (Super Like with the note),
        ``'unavailable'`` (Tinder offered to sell Super Likes instead; the upsell was dismissed and
        nothing was sent) or ``'failed'`` (the control could not be pressed). The caller still has
        to confirm the card advanced, exactly as for like/nope.
        """
        if not self._press("super_like", "Enter"):
            return "failed"
        self._sleep(lognormal_delay(0.8, 1.8, self.rng))
        text = self._dialog_text()
        if text and self.is_super_like_upsell(text):
            self.dismiss_popups()
            return "unavailable"
        box = self._first("note_input")
        if box is None:
            return "sent"
        typed = False
        if note:
            try:
                self.mouse.click(box)
                self._sleep(lognormal_delay(0.3, 0.9, self.rng))
                self.type_like_human(note)
                typed = True
                self._sleep(lognormal_delay(0.5, 1.5, self.rng))
            except Exception:
                typed = False
        send = self._button_by_text(NOTE_SEND_TEXTS)
        if send is None:
            return "failed"
        self.mouse.click(send)
        self._sleep(lognormal_delay(0.8, 1.6, self.rng))
        text = self._dialog_text()
        if text and self.is_super_like_upsell(text):
            self.dismiss_popups()
            return "unavailable"
        return "sent_note" if typed else "sent"

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
            loc = self._button_by_text([text])
            if loc is None:
                continue
            try:
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
