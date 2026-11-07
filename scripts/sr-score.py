#!/usr/bin/env python3
"""Score windows of 8 full frames with a frozen d3-nano, with or without a 2x bicubic up-scale (2026-11-06,
full-frame-sr proposal, decisions 1 and 5).

  uv run --frozen python scripts/sr-score.py --frames-dir <frames dir> --index <windows.jsonl> --upscale bicubic2 \\
      --out <dir>/B/scores.jsonl [--limit-windows N]

Conditions: B = original frames + --upscale bicubic2; C = SR frames (scripts/sr-frames.py) + --upscale none.
Original frames + --upscale none is the draft check: it must reproduce the saved zero-shot scores.
One scores.jsonl line per window: video, window, start_s, end_s, label, score = p(Yes), hot_area, seconds, size.
Resumable: skips windows already in --out. Writes config.yaml next to --out.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import s1stream.plugins  # noqa: E402,F401
from s1stream.assets import resolve  # noqa: E402
from s1stream.prompt import PROMPTS  # noqa: E402
from s1stream.registry import BACKBONES  # noqa: E402
from s1stream.types import ModelInput, Question  # noqa: E402


def keep(frame: np.ndarray) -> np.ndarray:
    return frame


def bicubic2(frame: np.ndarray) -> np.ndarray:
    """2x per side with cubic interpolation (the same call as the classmate's crop up-scale)."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (2 * w, 2 * h), interpolation=cv2.INTER_CUBIC)


UPSCALE = {"none": keep, "bicubic2": bicubic2}


def window_images(frames: np.ndarray, upscale: str) -> list[Image.Image]:
    """The model input of one window: frames [8, H, W, 3], each passed through UPSCALE[upscale]."""
    resize = UPSCALE[upscale]
    return [Image.fromarray(resize(np.ascontiguousarray(f))) for f in frames]


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
    ap.add_argument("--upscale", choices=sorted(UPSCALE), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-windows", type=int, default=None)
    ap.add_argument("--asset", default="d3-nano")
    ap.add_argument("--memory-cap-gb", type=float, default=40)
    ap.add_argument("--adapter", default=None, help="fine-tuned run dir (adapter/, readout.safetensors, calibration.json)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.index.read_text().splitlines()]
    if args.limit_windows is not None:
        rows = rows[:args.limit_windows]
    done = read_done(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) for k, v in vars(args).items()} | {"torch": str(torch.__version__),
                                                           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    (args.out.parent / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    p = PROMPTS["v1"]
    question = Question(state=p["state"], text=p["text"], options=tuple(p["options"]), positive=p["positive"])
    positive = question.options.index(question.positive)
    t_load = time.perf_counter()
    spec = {"name": "d3", "asset": args.asset, "path": str(resolve(args.asset)), "memory_cap_gb": args.memory_cap_gb}
    if args.adapter:
        spec["adapter"] = args.adapter
    backbone = BACKBONES.build(spec)
    print(f"loaded {args.asset} in {time.perf_counter() - t_load:.1f} s; upscale {args.upscale}; "
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
            images = window_images(frames[r["window"]], args.upscale)
            t = time.perf_counter()
            with torch.inference_mode():
                logits = backbone.forward(ModelInput(images=images, text=None), question).logits
            score = float(logits.detach().float().cpu().softmax(dim=-1)[positive])
            seconds = time.perf_counter() - t
            out.write(json.dumps({"video": r["video"], "window": r["window"], "start_s": r["start_s"],
                                  "end_s": r["end_s"], "label": r["label"], "score": score,
                                  "hot_area": r["hot_area"], "seconds": seconds,
                                  "size": list(images[0].size)}) + "\n")
            n += 1
            if n % 500 == 0:
                out.flush()
                print(f"{n} windows, {(time.perf_counter() - t0) / n:.3f} s/window", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"done: {n} windows in {time.perf_counter() - t0:.0f} s; peak GPU {peak:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
