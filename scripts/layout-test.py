#!/usr/bin/env python3
"""Frame-layout test (2026-11-05): how a frozen d3-nano reads full frames and TGM crops arranged in different orders.
Reads the frames saved by scripts/extract-frames.py, so every layout sees the same pixels.

  uv run --frozen python scripts/layout-test.py --frames-dir <frames run> --index <windows.jsonl> --layout A \\
      --out <dir>/A/scores.jsonl [--limit-windows N]

F<k> = frame k of the window (8 frames, 0.25 s apart); C<k> = crop of frame k: the window's TGM 1/3 box
(box_third in windows.jsonl), up-scaled 2x with cubic interpolation, the same crop as input_forms/foveated.py.
  draft  F0 F1 F2 F3 F4 F5 F6 F7           8 images (check against the saved zero-shot run)
  A      F0 C1 F2 C3 F4 C5 F6 C7           8 images, interleaved (the fov1 format)
  B      F0 C0 F2 C2 F4 C4 F6 C6           8 images, each crop next to its own full frame, 4 time points
  C      F0 C0 F1 C1 ... F7 C7            16 images, paired, 8 time points
Every window is cropped (no fov2 area gate); the gate is applied offline, where a skipped crop means the draft score.
One scores.jsonl line per window: video, window, start_s, end_s, label, score = p(Yes), hot_area, seconds.
Resumable: skips windows already in --out.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import s1stream.plugins  # noqa: E402,F401
from s1stream.assets import resolve  # noqa: E402
from s1stream.prompt import PROMPTS  # noqa: E402
from s1stream.registry import BACKBONES  # noqa: E402
from s1stream.types import ModelInput, Question  # noqa: E402

UPSCALE = 2
# Each layout is a list of (kind, frame) slots; kind "F" = full frame, "C" = crop.
LAYOUTS = {
    "draft": [("F", k) for k in range(8)],
    "A": [("F" if k % 2 == 0 else "C", k) for k in range(8)],
    "B": [slot for k in (0, 2, 4, 6) for slot in (("F", k), ("C", k))],
    "C": [slot for k in range(8) for slot in (("F", k), ("C", k))],
}


def crop(frame: np.ndarray, box: list[int]) -> np.ndarray:
    """The box (y0, x0, h, w) of one frame, up-scaled UPSCALE x with cubic interpolation."""
    y0, x0, h, w = box
    return cv2.resize(frame[y0:y0 + h, x0:x0 + w], (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC)


def layout_images(frames: np.ndarray, box: list[int], layout: str) -> list[Image.Image]:
    """The model input of one window: frames [8, H, W, 3] arranged as LAYOUTS[layout]."""
    images = []
    for kind, k in LAYOUTS[layout]:
        if kind == "F":
            images.append(Image.fromarray(np.ascontiguousarray(frames[k])))
        else:
            images.append(Image.fromarray(crop(np.ascontiguousarray(frames[k]), box)))
    return images


def read_done(path: Path) -> set[tuple[str, int]]:
    """(video, window) pairs already scored in path."""
    if not path.exists():
        return set()
    return {(r["video"], r["window"]) for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--layout", choices=sorted(LAYOUTS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-windows", type=int, default=None)
    ap.add_argument("--asset", default="d3-nano")
    ap.add_argument("--memory-cap-gb", type=float, default=40)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.index.read_text().splitlines()]
    if args.limit_windows is not None:
        rows = rows[:args.limit_windows]
    done = read_done(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    p = PROMPTS["v1"]
    question = Question(state=p["state"], text=p["text"], options=tuple(p["options"]), positive=p["positive"])
    positive = question.options.index(question.positive)
    t_load = time.perf_counter()
    backbone = BACKBONES.build({"name": "d3", "asset": args.asset, "path": str(resolve(args.asset)),
                                "memory_cap_gb": args.memory_cap_gb})
    print(f"loaded {args.asset} in {time.perf_counter() - t_load:.1f} s; layout {args.layout}; "
          f"{len(rows) - len(done)} windows to score", flush=True)

    video, frames = None, None
    n, t0 = 0, time.perf_counter()
    with args.out.open("a") as out:
        for r in rows:
            if (r["video"], r["window"]) in done:
                continue
            if r["video"] != video:
                video = r["video"]
                frames = np.load(args.frames_dir / f"{video}.npy", mmap_mode="r")
            images = layout_images(frames[r["window"]], r["box_third"], args.layout)
            t = time.perf_counter()
            with torch.inference_mode():
                logits = backbone.forward(ModelInput(images=images, text=None), question).logits
            score = float(logits.softmax(dim=-1)[positive])
            seconds = time.perf_counter() - t
            out.write(json.dumps({"video": r["video"], "window": r["window"], "start_s": r["start_s"],
                                  "end_s": r["end_s"], "label": r["label"], "score": score,
                                  "hot_area": r["hot_area"], "seconds": seconds}) + "\n")
            n += 1
            if n % 500 == 0:
                out.flush()
                print(f"{n} windows, {(time.perf_counter() - t0) / n:.3f} s/window", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"done: {n} windows in {time.perf_counter() - t0:.0f} s; peak GPU {peak:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
