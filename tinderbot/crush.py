"""Crush policy: when to spend a Super Like, when to attach a note, and the daily rations.

A *crush* is a profile whose final probability clears ``crush.super_like_threshold``; it gets the
day's Super Like. A *super crush* clears ``crush.message_threshold`` as well and gets the Super Like
with a short note (Tinder only lets you message before matching by attaching a note to a Super Like).
Both are rationed per local day (default one each), and once Tinder reports no Super Like available
the bot stops trying until the next day.
"""

from __future__ import annotations

import datetime as dt
import random

from .browser.pacing import local_midnight_ts
from .config import CrushConfig
from .likeness.scorer import Verdict
from .storage import ProfileRecord, Storage

EVENT_SUPERLIKE = "superlike"
EVENT_MESSAGE = "crush_message"
META_UNAVAILABLE = "superlike_unavailable_day"


def _today() -> str:
    return dt.date.today().isoformat()


class CrushPolicy:
    def __init__(self, cfg: CrushConfig, storage: Storage, rng: random.Random | None = None):
        self.cfg = cfg
        self.storage = storage
        self.rng = rng or random.Random()

    # ---- budgets ------------------------------------------------------------------
    def super_likes_today(self) -> int:
        return self.storage.count_events(EVENT_SUPERLIKE, local_midnight_ts())

    def messages_today(self) -> int:
        return self.storage.count_events(EVENT_MESSAGE, local_midnight_ts())

    def unavailable_today(self) -> bool:
        return self.storage.get_meta(META_UNAVAILABLE) == _today()

    def mark_unavailable(self) -> None:
        """Tinder said there is no Super Like left (upsell shown): stop trying for the day."""
        self.storage.set_meta(META_UNAVAILABLE, _today())

    def can_super_like(self, remaining_reported: int | None = None) -> bool:
        if not self.cfg.enabled or self.unavailable_today():
            return False
        if remaining_reported is not None and remaining_reported <= 0:
            return False
        return self.super_likes_today() < self.cfg.max_super_likes_per_day

    def can_message(self) -> bool:
        return bool(self.cfg.messages) and self.messages_today() < self.cfg.max_messages_per_day

    # ---- decision -----------------------------------------------------------------
    def plan(self, verdict: Verdict, remaining_reported: int | None = None) -> str | None:
        """'superlike_note' | 'superlike' | None for this verdict, given today's rations."""
        if not verdict.like or not verdict.crush or not self.can_super_like(remaining_reported):
            return None
        if verdict.super_crush and self.can_message():
            return "superlike_note"
        return "superlike"

    def note_for(self, profile: ProfileRecord) -> str:
        template = self.rng.choice(self.cfg.messages)
        name = (profile.name or "").strip().split(" ")[0]
        note = template.replace("{name}", name) if name else template.replace(" {name}", "").replace("{name}", "")
        return " ".join(note.split())[:140]

    def record(self, profile_id: str, note: str | None) -> None:
        """Log a confirmed Super Like (and its note) so the daily counters see it."""
        self.storage.log_event(EVENT_SUPERLIKE, {"profile_id": profile_id, "note": bool(note)})
        if note:
            self.storage.log_event(EVENT_MESSAGE, {"profile_id": profile_id, "note": note})
