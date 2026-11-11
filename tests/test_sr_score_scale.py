"""scripts/sr-score.py --upscale <method><factor> (2026-11-10): scale and interpolation sweep."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sr_score_scale", ROOT / "scripts/sr-score.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)

FRAMES = np.random.default_rng(2).integers(0, 256, (8, 240, 320, 3), dtype=np.uint8)
SIZES = {"1.25": (400, 300), "1.5": (480, 360), "1.75": (560, 420), "2": (640, 480), "3": (960, 720)}


@pytest.mark.parametrize("factor", sorted(SIZES))
@pytest.mark.parametrize("method", sorted(sr.INTERPOLATION))
def test_size_and_interpolation(method, factor):
    images = sr.window_images(FRAMES, f"{method}{factor}")
    expected = cv2.resize(np.ascontiguousarray(FRAMES[3]), SIZES[factor], interpolation=sr.INTERPOLATION[method])
    assert all(image.size == SIZES[factor] for image in images)
    assert np.array_equal(np.asarray(images[3]), expected)


def test_bicubic2_is_the_old_function():
    assert sr.UPSCALE["bicubic2"] is sr.bicubic2
    assert np.array_equal(sr.UPSCALE["bicubic2"](FRAMES[0]), sr.resizer("bicubic", "2")(FRAMES[0]))


def test_methods_differ():
    outputs = [np.asarray(sr.window_images(FRAMES[:1], f"{m}2")[0]) for m in sr.INTERPOLATION]
    assert len({o.tobytes() for o in outputs}) == len(outputs)
