import math
import random

from tinderbot.browser.humanize import ReadingModel, click_point, lognormal_delay, mouse_path


def test_mouse_path_ends_at_target_and_is_curved():
    rng = random.Random(1)
    pts = mouse_path((100, 100), (700, 400), rng, overshoot=False)
    assert pts[-1][:2] == (700, 400)
    assert 8 <= len(pts) <= 64
    # waypoints should not all lie on the straight line (a human curve)
    dev = 0.0
    for x, y, _ in pts:
        # distance to the line through start/end
        dev = max(dev, abs((400 - 100) * x - (700 - 100) * y + 700 * 100 - 400 * 100) / math.hypot(600, 300))
    assert dev > 3
    # timing is positive and plausibly sub-second-ish overall
    total = sum(dt for _, _, dt in pts)
    assert 0.15 < total < 1.5
    assert all(dt > 0 for _, _, dt in pts)


def test_mouse_path_overshoot_settles_on_target():
    rng = random.Random(3)
    pts = mouse_path((0, 0), (600, 0), rng, overshoot=True)
    assert pts[-1][:2] == (600, 0)
    assert max(x for x, _, _ in pts) > 600  # went past, then came back


def test_short_moves_and_zero_moves():
    assert mouse_path((5, 5), (5, 5)) == [(5, 5, 0.01)]
    pts = mouse_path((10, 10), (14, 12), random.Random(0))
    assert pts[-1][:2] == (14, 12)


def test_click_point_inside_box():
    rng = random.Random(0)
    box = {"x": 100, "y": 200, "width": 80, "height": 40}
    for _ in range(200):
        x, y = click_point(box, rng)
        assert 112 <= x <= 168 and 206 <= y <= 234


def test_lognormal_delay_bounds():
    rng = random.Random(0)
    vals = [lognormal_delay(0.5, 2.0, rng) for _ in range(500)]
    assert all(0.5 <= v <= 2.0 for v in vals)
    assert lognormal_delay(1.0, 1.0) == 1.0
    m = sum(vals) / len(vals)
    assert 0.8 < m < 1.6


def test_reading_model_scales_with_content():
    rm = ReadingModel((1, 2), (1, 1), 0.01)
    rng = random.Random(0)
    short = rm.seconds(0, 0, rng)
    long_ = rm.seconds(3, 400, rng)
    assert 1 <= short <= 2
    assert long_ > short + 3
