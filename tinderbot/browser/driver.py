"""Launch a persistent, headed, real-Chrome context (Patchright if installed, Playwright otherwise).

Why this shape (see docs/RESEARCH.md):
  * persistent ``user_data_dir``  -> cookies, localStorage and the Arkose device token survive restarts,
                                     so you log in once and the session ages like a normal user's
  * ``channel="chrome"``          -> real Google Chrome build, not the automation-flavoured Chromium
  * headed + ``no_viewport``      -> headless and fixed 1280x720 viewports are classic bot tells
  * no UA / header / fingerprint spoofing: inconsistencies are what risk engines score, not defaults
  * Patchright removes the CDP ``Runtime.enable`` / ``Console.enable`` leaks and automation flags
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from ..config import Config


def _api():
    try:
        from patchright.sync_api import sync_playwright  # type: ignore

        return sync_playwright, "patchright"
    except ImportError:
        from playwright.sync_api import sync_playwright

        return sync_playwright, "playwright"


@dataclass
class BrowserSession:
    playwright: Any
    context: Any
    page: Any
    backend: str

    def close(self) -> None:
        try:
            self.context.close()
        finally:
            self.playwright.stop()


def launch(cfg: Config, url: str | None = None) -> BrowserSession:
    cfg.ensure_dirs()
    sync_playwright, backend = _api()
    if backend == "playwright":
        print("note: 'patchright' is not installed; using plain Playwright (more detectable). "
              "pip install patchright", file=sys.stderr)
    pw = sync_playwright().start()
    kwargs: dict[str, Any] = dict(
        user_data_dir=str(cfg.profile_path),
        headless=cfg.browser.headless,
        no_viewport=True,
        slow_mo=cfg.browser.slow_mo_ms or 0,
        ignore_default_args=["--enable-automation"],
        args=["--disable-blink-features=AutomationControlled", "--no-first-run", "--no-default-browser-check"],
    )
    if cfg.browser.executable_path:
        kwargs["executable_path"] = cfg.browser.executable_path
    elif cfg.browser.channel:
        kwargs["channel"] = cfg.browser.channel
    if cfg.browser.locale:
        kwargs["locale"] = cfg.browser.locale
    try:
        context = pw.chromium.launch_persistent_context(**kwargs)
    except Exception as e:
        if "channel" in kwargs:
            print(f"note: could not launch channel={kwargs['channel']} ({str(e).splitlines()[0]}); "
                  "falling back to bundled Chromium", file=sys.stderr)
            kwargs.pop("channel")
            context = pw.chromium.launch_persistent_context(**kwargs)
        else:
            pw.stop()
            raise
    page = context.pages[0] if context.pages else context.new_page()
    if url:
        page.goto(url, wait_until="domcontentloaded")
    return BrowserSession(pw, context, page, backend)
