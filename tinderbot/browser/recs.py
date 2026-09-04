"""Parse Tinder's own recommendations JSON (``/v2/recs/core`` responses the web app fetches anyway).

We never call the API ourselves: a response listener on the page reads what the app already loaded,
which gives clean photo URLs, bio, age, distance, interests and the verified badge without scraping.
The parser is defensive because field names drift between API versions.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from ..storage import ProfileRecord

RECS_URL_PATTERN = re.compile(r"/v\d+/recs/core|/recs/core")


def _age_from_birth_date(s: str | None) -> int | None:
    if not s:
        return None
    try:
        born = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            born = dt.datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None
    today = dt.date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _best_photo_url(photo: dict[str, Any], prefer_width: int = 640) -> str | None:
    files = photo.get("processedFiles") or photo.get("processed_files") or []
    best, best_d = None, 10 ** 9
    for f in files:
        url, w = f.get("url"), f.get("width") or 0
        if url and abs(w - prefer_width) < best_d:
            best, best_d = url, abs(w - prefer_width)
    return best or photo.get("url")


def _names(items: Iterable[Any], *keys: str) -> list[str]:
    out: list[str] = []
    for it in items or []:
        if isinstance(it, str):
            out.append(it)
            continue
        if not isinstance(it, dict):
            continue
        for k in keys:
            v = it.get(k)
            if isinstance(v, dict):
                v = v.get("name")
            if v:
                out.append(str(v))
                break
    return out


def parse_rec(item: dict[str, Any]) -> ProfileRecord | None:
    user = item.get("user") if isinstance(item.get("user"), dict) else item
    uid = user.get("_id") or user.get("id")
    if not uid:
        return None
    photos = [u for u in (_best_photo_url(p) for p in user.get("photos") or [] if isinstance(p, dict)) if u]
    badges = user.get("badges") or []
    verified = any((b.get("type") if isinstance(b, dict) else b) == "selfie_verified" for b in badges) \
        or bool(user.get("is_verified"))
    interests: list[str] = []
    exp = item.get("experiment_info") or {}
    ui = (exp.get("user_interests") or {}).get("selected_interests") or user.get("user_interests") or []
    interests = _names(ui, "name")
    dist_mi = item.get("distance_mi") or user.get("distance_mi")
    dist_km = float(dist_mi) * 1.609344 if dist_mi is not None else None
    return ProfileRecord(
        id=str(uid),
        name=str(user.get("name") or ""),
        age=_age_from_birth_date(user.get("birth_date")) or user.get("age"),
        bio=str(user.get("bio") or ""),
        distance_km=dist_km,
        verified=verified,
        jobs=_names(user.get("jobs") or [], "title", "company"),
        schools=_names(user.get("schools") or [], "name"),
        interests=interests,
        photo_urls=photos,
        raw=item,
        s_number=item.get("s_number"),
    )


def parse_recs_payload(payload: str | bytes | dict) -> list[ProfileRecord]:
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return []
    out = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in (None, "user"):
            continue  # ads / boosts / "fast match" teasers
        rec = parse_rec(item)
        if rec:
            out.append(rec)
    return out


def dom_profile_id(name: str, age: int | None, first_photo_url: str | None) -> str:
    """Stable fallback id when the recs JSON was not observed (built from what the card shows)."""
    key = f"{name}|{age}|{first_photo_url or ''}"
    return "dom_" + hashlib.sha1(key.encode()).hexdigest()[:20]


class RecsQueue:
    """Profiles observed from the network, in arrival order, looked up by (name, age) or first photo."""

    def __init__(self, max_size: int = 500):
        self._by_id: OrderedDict[str, ProfileRecord] = OrderedDict()
        self.max_size = max_size
        self.seen_payloads = 0

    def add_payload(self, payload: str | bytes | dict) -> int:
        recs = parse_recs_payload(payload)
        for r in recs:
            self._by_id[r.id] = r
            self._by_id.move_to_end(r.id)
        while len(self._by_id) > self.max_size:
            self._by_id.popitem(last=False)
        self.seen_payloads += 1
        return len(recs)

    def __len__(self) -> int:
        return len(self._by_id)

    def match(self, name: str | None, age: int | None, photo_urls: Iterable[str] = ()) -> ProfileRecord | None:
        urls = {u.split("?")[0] for u in photo_urls if u}
        cands = list(self._by_id.values())
        if name:
            n = name.strip().lower()
            matches = [r for r in cands if r.name.strip().lower() == n and (age is None or r.age is None or r.age == age)]
            if len(matches) == 1:
                return matches[0]
            if matches and urls:
                for r in matches:
                    if any(u.split("?")[0] in urls for u in r.photo_urls):
                        return r
            if matches:
                return matches[0]
        # The DOM can expose background images from stacked cards.  Treat a
        # photo-only match as a fallback, never as stronger evidence than the
        # visible name and age.
        if urls:
            for r in cands:
                if any(u.split("?")[0] in urls for u in r.photo_urls):
                    return r
        return None

    def pop(self, profile_id: str) -> None:
        self._by_id.pop(profile_id, None)
