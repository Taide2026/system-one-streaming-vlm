#!/usr/bin/env python3
"""Extract the 8 frames of every window once (2026-11-05), so every layout test (interleaved / paired / PiP) reads
the same pixels. No model - OpenCV, CPU.

  uv run --frozen python scripts/extract-frames.py --config configs/zs_ucf_v1_d3nano_eval.yaml \\
      --frames-dir $S1_FEATURES/frames/frames_ucf290_f8/<run> --out results/spark/frames_ucf290_f8/<run> [--limit-videos N]

Videos and windows come from the config's dataset and window sections (the same 290 videos and plan_windows as an
eval run). Frames are read with input_forms.frames.read_frames, so the pixels equal what the runner feeds the model.
Outputs:
  <frames-dir>/<video>.npy   uint8 [n_windows, 8, H, W, 3], RGB (large, not in git)
  <out>/config.yaml          the config used
  <out>/manifest.json        the selected videos
  <out>/windows.jsonl        one line per window: video, window, start_s, end_s, label, frame_indices,
                             box_third / box_half = TGM box (y0, x0, h, w) at 1/3 and 1/2 of the frame,
                             hot_area = share of TGM pixels above half the peak (fov2 skips the crop at >= 0.045)
Crops are not stored: they follow from the frames and the box. Skips videos whose .npy already exists (resumable).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import s1stream.plugins  # noqa: E402,F401
from s1stream.config import load_config  # noqa: E402
from s1stream.input_forms.foveated import tgm_box  # noqa: E402
from s1stream.input_forms.frames import read_frames  # noqa: E402
from s1stream.registry import DATASETS  # noqa: E402
from s1stream.windows import plan_windows  # noqa: E402


def hot_area(arrays: list[np.ndarray]) -> float:
    """Share of TGM pixels above half the peak, the same heat map as foveate()'s max_area check."""
    grey = [cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32) for a in arrays]
    heat = cv2.GaussianBlur(np.sum([np.abs(x - y) for x, y in zip(grey[1:], grey[:-1])], axis=0), (31, 31), 0)
    peak = float(heat.max())
    if peak <= 0:
        return 0.0
    return float((heat > 0.5 * peak).mean())


def window_row(window, arrays: list[np.ndarray]) -> dict:
    """The index line of one window."""
    return {"video": window.video, "window": window.index, "start_s": window.start_s, "end_s": window.end_s,
            "label": window.label, "frame_indices": list(window.frame_indices),
            "box_third": [int(v) for v in tgm_box(arrays, 1 / 3)], "box_half": [int(v) for v in tgm_box(arrays, 1 / 2)],
            "hot_area": hot_area(arrays)}


def extract_video(item, config, frames_dir: Path) -> list[dict]:
    """Read every window of one video, save its frames as one .npy, return its index lines."""
    windows = plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames)
    stack, rows = [], []
    for window in windows:
        arrays = [np.asarray(im) for im in read_frames(item.path, window.frame_indices)]
        stack.append(np.stack(arrays))
        rows.append(window_row(window, arrays))
    np.save(frames_dir / f"{item.name}.npy", np.stack(stack))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-videos", type=int, default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    items = DATASETS.build(config.dataset).items()
    if args.limit_videos is not None:
        items = items[:args.limit_videos]
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, args.out / "config.yaml")
    manifest = [{"name": i.name, "category": i.category, "fps": i.fps, "n_frames": i.n_frames,
                 "events": [list(e) for e in i.events]} for i in items]
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    index_path = args.out / "windows.jsonl"
    done = {json.loads(line)["video"] for line in index_path.open()} if index_path.exists() else set()
    with index_path.open("a") as index:
        for n, item in enumerate(items, 1):
            if item.name in done and (args.frames_dir / f"{item.name}.npy").exists():
                continue
            t0 = time.perf_counter()
            rows = extract_video(item, config, args.frames_dir)
            index.write("".join(json.dumps(r) + "\n" for r in rows))
            index.flush()
            print(f"[{n}/{len(items)}] {item.name}: {len(rows)} windows, {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
