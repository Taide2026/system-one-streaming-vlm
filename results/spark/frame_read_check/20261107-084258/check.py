#!/usr/bin/env python3
"""Check the sequential read_frames against the saved frames of the old read_frames (2026-11-07). CPU only.

  uv run --frozen python results/spark/frame_read_check/<run>/check.py --config <run>/config.yaml \\
      --frames-dir $S1_FEATURES/frames/frames_ucf290_f8/20261105-180529 --out-dir <run>

The saved frames come from scripts/extract-frames.py at e0c2a1b, which used the old read_frames (one seek per frame).
Three passes, one check-log.jsonl line per video and pass:
  sequential  all videos, windows in order (the runner's access pattern): new read_frames vs the saved frames
  shuffled    --shuffled-videos videos, windows in a random order (training's access pattern): new vs saved
  old         --old-videos videos, windows in order, the old seek-per-frame reader: old vs saved, and its time
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
import s1stream.plugins  # noqa: E402,F401
from s1stream.config import load_config  # noqa: E402
from s1stream.input_forms import frames as frames_module  # noqa: E402
from s1stream.registry import DATASETS  # noqa: E402
from s1stream.windows import plan_windows  # noqa: E402


def old_read_frames(path: Path, indices) -> list[np.ndarray]:
    """The read_frames before 2026-11-07: open the video, one seek per frame, release."""
    cap = cv2.VideoCapture(str(path))
    try:
        out = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise frames_module.FrameReadError(f"Could not read {path} at frame {index}")
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return out
    finally:
        cap.release()


def new_read_frames(path: Path, indices) -> list[np.ndarray]:
    return [np.asarray(im) for im in frames_module.read_frames(path, indices)]


READERS = {"sequential": new_read_frames, "shuffled": new_read_frames, "old": old_read_frames}


def check_video(item, windows, saved: np.ndarray, reader, order: list[int]) -> dict:
    """Read the windows in `order` with reader; count windows that differ from the saved frames."""
    differ, max_diff, seconds = 0, 0, 0.0
    for w in order:
        start = time.perf_counter()
        arrays = reader(item.path, windows[w].frame_indices)
        seconds += time.perf_counter() - start
        diff = int(np.abs(np.stack(arrays).astype(np.int16) - saved[w].astype(np.int16)).max())
        differ += diff > 0
        max_diff = max(max_diff, diff)
    return {"video": item.name, "windows": len(order), "differ": differ, "max_abs_diff": max_diff,
            "ms_per_window": seconds / len(order) * 1000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shuffled-videos", type=int, default=30)
    ap.add_argument("--old-videos", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()

    config = load_config(args.config)
    items = DATASETS.build(config.dataset).items()
    rng = random.Random(args.seed)
    picks = {"sequential": items,
             "shuffled": rng.sample(items, args.shuffled_videos),
             "old": rng.sample(items, args.old_videos)}
    summary = {}
    with (args.out_dir / "check-log.jsonl").open("w") as log:
        for name, chosen in picks.items():
            rows = []
            for item in chosen:
                windows = plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames)
                saved = np.load(args.frames_dir / f"{item.name}.npy", mmap_mode="r")
                if saved.shape[0] != len(windows):
                    raise ValueError(f"{item.name}: {saved.shape[0]} saved windows, {len(windows)} planned")
                order = list(range(len(windows)))
                if name == "shuffled":
                    rng.shuffle(order)
                row = {"pass": name} | check_video(item, windows, saved, READERS[name], order)
                log.write(json.dumps(row) + "\n")
                log.flush()
                rows.append(row)
            frames_module.close_reader()
            n = sum(r["windows"] for r in rows)
            summary[name] = {"videos": len(rows), "windows": n, "windows_differ": sum(r["differ"] for r in rows),
                             "max_abs_diff": max(r["max_abs_diff"] for r in rows),
                             "ms_per_window": sum(r["ms_per_window"] * r["windows"] for r in rows) / n}
            print(name, json.dumps(summary[name]), flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
