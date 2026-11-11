"""scripts/sr-score.py --upscale odd-bicubic2 (2026-11-09): equal to the fovfull2x_d3nano_c training input."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from s1stream.input_forms.foveated import foveate  # noqa: E402


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sr = load("sr_score", ROOT / "scripts/sr-score.py")


def test_odd_bicubic2_equals_foveate_full_frame():
    frames = np.random.default_rng(0).integers(0, 256, (8, 240, 320, 3), dtype=np.uint8)
    ours = sr.window_images(frames, "odd-bicubic2")
    training = foveate([Image.fromarray(frame) for frame in frames], frac=1.0, upscale=2)
    for index, (a, b) in enumerate(zip(ours, training)):
        assert a.size == ((640, 480) if index % 2 == 1 else (320, 240))
        assert np.array_equal(np.asarray(a), np.asarray(b.convert("RGB")))


def test_old_choices_unchanged():
    frames = np.random.default_rng(1).integers(0, 256, (8, 24, 32, 3), dtype=np.uint8)
    assert all(np.array_equal(np.asarray(image), frame) for image, frame in zip(sr.window_images(frames, "none"), frames))
    assert all(image.size == (64, 48) for image in sr.window_images(frames, "bicubic2"))
