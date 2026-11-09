"""scripts/train-floor.py helpers of floor2x_hn (2026-11-09): stratified 2x choice, bicubic 2x, frame cache."""
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from s1stream.traindata import Record  # noqa: E402


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tf = load("train_floor", ROOT / "scripts/train-floor.py")


def record(i: int, y: float, category: str) -> Record:
    return Record(video=f"v{i}", path=f"/x/v{i}.mp4", category=category, window=i, start_s=2.0 * i,
                  end_s=2.0 * i + 2, frame_indices=tuple(range(8)), y=y)


def test_pick_upscaled_takes_the_share_of_each_group():
    records = ([record(i, 1.0, "Abuse") for i in range(100)] + [record(100 + i, 0.0, "Abuse") for i in range(50)]
               + [record(150 + i, 0.0, "Normal") for i in range(200)])
    chosen = tf.pick_upscaled(records, 0.3, 0)
    assert sum(i < 100 for i in chosen) == 30
    assert sum(100 <= i < 150 for i in chosen) == 15
    assert sum(i >= 150 for i in chosen) == 60
    assert chosen == tf.pick_upscaled(records, 0.3, 0)  # same seed, same choice


def test_bicubic2_equals_sr_score():
    a = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    out = np.asarray(tf.bicubic2(Image.fromarray(a)))
    assert out.shape == (480, 640, 3)
    assert np.array_equal(out, cv2.resize(a, (640, 480), interpolation=cv2.INTER_CUBIC))


def test_frames_cache_round_trip(tmp_path):
    frames = np.random.default_rng(1).integers(0, 256, (8, 24, 32, 3), dtype=np.uint8)
    r = record(3, 0.0, "Normal")
    np.save(tmp_path / tf.cache_name(r), frames)
    ds = tf.FloorDataset([], None, frames_cache=tmp_path)
    images = ds.images_of(r)
    assert np.array_equal(np.stack([np.asarray(im) for im in images]), frames)
