"""Human-like input: Bezier mouse paths, Fitts-law timing, jitter, overshoot, log-normal pauses.

The path generators are pure functions (unit-tested); :class:`HumanMouse` drives a Playwright page.
Arkose/Tinder score mouse telemetry, so straight-line teleporting clicks are the first thing to avoid.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass


def lognormal_delay(lo: float, hi: float, rng: random.Random | None = None) -> float:
    """Random delay in [lo, hi] with a right-skewed (log-normal) shape, like human reaction times."""
    rng = rng or random
    if hi <= lo:
        return lo
    mu = math.log((lo + hi) / 2.0)
    for _ in range(8):
        v = rng.lognormvariate(mu, 0.35)
        if lo <= v <= hi:
            return v
    return rng.uniform(lo, hi)


def _bezier(p0, p1, p2, p3, t: float) -> tuple[float, float]:
    u = 1 - t
    x = u ** 3 * p0[0] + 3 * u ** 2 * t * p1[0] + 3 * u * t ** 2 * p2[0] + t ** 3 * p3[0]
    y = u ** 3 * p0[1] + 3 * u ** 2 * t * p1[1] + 3 * u * t ** 2 * p2[1] + t ** 3 * p3[1]
    return x, y


def _ease(t: float) -> float:
    """Minimum-jerk style profile: slow start, fast middle, slow precise landing."""
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def mouse_path(start: tuple[float, float], end: tuple[float, float], rng: random.Random | None = None,
               overshoot: bool | None = None) -> list[tuple[float, float, float]]:
    """Return a list of (x, y, dt_seconds) waypoints from start to end.

    * cubic Bezier with control points offset perpendicular to the travel direction
    * step count from the distance (Fitts-like), duration 250-900 ms for typical screen moves
    * small Gaussian jitter that fades out near the target
    * optional overshoot past the target followed by a short correction
    """
    rng = rng or random
    sx, sy = start
    ex, ey = end
    dist = math.hypot(ex - sx, ey - sy)
    if dist < 1:
        return [(ex, ey, 0.01)]
    # perpendicular unit vector for control point offsets
    px, py = -(ey - sy) / dist, (ex - sx) / dist
    bend = rng.gauss(0, 0.18) * dist
    c1 = (sx + (ex - sx) * rng.uniform(0.2, 0.4) + px * bend, sy + (ey - sy) * rng.uniform(0.2, 0.4) + py * bend)
    c2 = (sx + (ex - sx) * rng.uniform(0.6, 0.8) + px * bend * rng.uniform(0.2, 0.8),
          sy + (ey - sy) * rng.uniform(0.6, 0.8) + py * bend * rng.uniform(0.2, 0.8))
    steps = int(max(8, min(60, 8 + dist / 14)))
    duration = max(0.18, min(1.1, 0.25 + 0.0009 * dist + rng.gauss(0, 0.05)))
    if overshoot is None:
        overshoot = dist > 150 and rng.random() < 0.18
    target = (ex, ey)
    if overshoot:
        ov = rng.uniform(4, 14)
        target = (ex + (ex - sx) / dist * ov, ey + (ey - sy) / dist * ov)
    pts: list[tuple[float, float, float]] = []
    prev_t = 0.0
    for i in range(1, steps + 1):
        t = _ease(i / steps)
        x, y = _bezier(start, c1, c2, target, t)
        fade = 1.0 - i / steps
        x += rng.gauss(0, 1.2) * fade
        y += rng.gauss(0, 1.2) * fade
        dt = max(0.004, (t - prev_t) * duration + rng.gauss(0, 0.002))
        prev_t = t
        pts.append((x, y, dt))
    if overshoot:  # settle back onto the real target
        for j in range(1, 5):
            f = j / 4
            pts.append((target[0] + (ex - target[0]) * f, target[1] + (ey - target[1]) * f, rng.uniform(0.012, 0.03)))
    pts[-1] = (ex, ey, pts[-1][2])
    return pts


def click_point(box: dict, rng: random.Random | None = None) -> tuple[float, float]:
    """Pick a click position inside a bounding box, Gaussian around the centre, never on the edge."""
    rng = rng or random
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    x = cx + rng.gauss(0, box["width"] * 0.12)
    y = cy + rng.gauss(0, box["height"] * 0.12)
    x = min(max(x, box["x"] + box["width"] * 0.15), box["x"] + box["width"] * 0.85)
    y = min(max(y, box["y"] + box["height"] * 0.15), box["y"] + box["height"] * 0.85)
    return x, y


@dataclass
class ReadingModel:
    base: tuple[float, float]
    per_photo: tuple[float, float]
    per_bio_char: float

    def seconds(self, photos_browsed: int, bio_len: int, rng: random.Random | None = None) -> float:
        rng = rng or random
        t = lognormal_delay(*self.base, rng)
        t += sum(lognormal_delay(*self.per_photo, rng) for _ in range(photos_browsed))
        t += min(bio_len, 400) * self.per_bio_char * rng.uniform(0.6, 1.4)
        return t


class HumanMouse:
    """Drive a Playwright ``Page`` mouse with generated paths. Tracks the virtual cursor position."""

    def __init__(self, page, rng: random.Random | None = None):
        self.page = page
        self.rng = rng or random.Random()
        vp = page.viewport_size or {"width": 1280, "height": 800}
        self.pos = (vp["width"] * self.rng.uniform(0.3, 0.7), vp["height"] * self.rng.uniform(0.3, 0.7))
        self._sleep = time.sleep

    def move_to(self, x: float, y: float) -> None:
        for px, py, dt in mouse_path(self.pos, (x, y), self.rng):
            self.page.mouse.move(px, py)
            self._sleep(dt)
        self.pos = (x, y)

    def click(self, locator, timeout: float = 5000) -> None:
        locator.wait_for(state="visible", timeout=timeout)
        box = locator.bounding_box()
        if box is None:
            raise RuntimeError("element has no bounding box")
        x, y = click_point(box, self.rng)
        self.move_to(x, y)
        self._sleep(lognormal_delay(0.05, 0.25, self.rng))
        self.page.mouse.down()
        self._sleep(lognormal_delay(0.04, 0.16, self.rng))
        self.page.mouse.up()
        self.pos = (x, y)

    def wiggle(self) -> None:
        """Tiny idle movements (people rarely hold the mouse perfectly still)."""
        x = self.pos[0] + self.rng.gauss(0, 12)
        y = self.pos[1] + self.rng.gauss(0, 8)
        self.move_to(max(2, x), max(2, y))

    def wander(self, width: int, height: int) -> None:
        """Move somewhere plausible (the card, the empty margin) as a person would while looking."""
        self.move_to(self.rng.uniform(width * 0.25, width * 0.75), self.rng.uniform(height * 0.2, height * 0.8))

    def scroll(self, dy: float) -> None:
        remaining = dy
        while abs(remaining) > 1:
            step = math.copysign(min(abs(remaining), self.rng.uniform(40, 140)), remaining)
            self.page.mouse.wheel(0, step)
            remaining -= step
            self._sleep(lognormal_delay(0.03, 0.12, self.rng))

    def pause(self, lo: float, hi: float) -> None:
        self._sleep(lognormal_delay(lo, hi, self.rng))
