#!/usr/bin/env python3
"""Super-resolve saved window frames 2x with Real-ESRGAN x2plus (2026-11-06, full-frame-sr proposal, decision 3).

  uv run --frozen python scripts/sr-frames.py --frames-dir <frames run> --index <windows.jsonl> --out-dir <dir> \\
      [--limit-windows N] [--memory-fraction 0.5]

Input: one <video>.npy per video from scripts/extract-frames.py, uint8 [windows, 8, 240, 320, 3].
Output: the same layout at 2x, uint8 [windows, 8, 480, 640, 3], lossless .npy. A video file is written as .tmp and
renamed when complete, so a rerun skips finished videos. --limit-windows keeps the first N index rows (smoke test).
Also writes <out-dir>/config.yaml and appends one sr-log.jsonl line per video (seconds, frames).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1stream.assets import resolve  # noqa: E402

WEIGHTS = "RealESRGAN_x2plus.pth"


def super_resolve(model, frames: np.ndarray, device: str, half: bool) -> np.ndarray:
    """uint8 [n, H, W, 3] -> uint8 [n, sH, sW, 3] through model (an image-to-image torch module, input in [0, 1])."""
    import torch

    x = torch.from_numpy(np.array(frames)).to(device).permute(0, 3, 1, 2)
    x = x.half() if half else x.float()
    with torch.inference_mode():
        y = model(x / 255.0)
    y = (y.float().clamp(0, 1) * 255.0).round().to(torch.uint8)
    return y.permute(0, 2, 3, 1).cpu().numpy()


def windows_per_video(index: Path, limit: int | None) -> dict[str, int]:
    """Video name -> number of leading windows to process, in index order."""
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    if limit is not None:
        rows = rows[:limit]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["video"]] = counts.get(r["video"], 0) + 1
    return counts


def load_model(device: str, half: bool):
    """Real-ESRGAN x2plus from the realesrgan-x2plus asset, loaded with spandrel, eval mode."""
    import spandrel

    loaded = spandrel.ModelLoader().load_from_file(str(resolve("realesrgan-x2plus") / WEIGHTS))
    if loaded.scale != 2:
        raise ValueError(f"expected a 2x model, got scale {loaded.scale}")
    model = loaded.model.eval().to(device)
    return model.half() if half else model


def sr_video(model, src: Path, dst: Path, windows: int, device: str, half: bool) -> int:
    """Super-resolve the first `windows` windows of one video file; returns the number of frames written."""
    frames = np.load(src, mmap_mode="r")[:windows]
    w, k, h, wd, c = frames.shape
    tmp = dst.with_suffix(".npy.tmp")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(w, k, 2 * h, 2 * wd, c))
    for i in range(w):
        out[i] = super_resolve(model, frames[i], device, half)
    out.flush()
    del out
    tmp.rename(dst)
    return w * k


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit-windows", type=int, default=None)
    ap.add_argument("--memory-fraction", type=float, default=0.5)
    ap.add_argument("--fp32", action="store_true", help="run SR in float32 (default: float16 on CUDA)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = device == "cuda" and not args.fp32
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) for k, v in vars(args).items()} | {"device": device, "half": half, "weights": WEIGHTS,
                                                           "torch": str(torch.__version__)}
    (args.out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    model = load_model(device, half)
    todo = windows_per_video(args.index, args.limit_windows)
    print(f"device {device}, half {half}; {len(todo)} videos", flush=True)
    t_all, n_all = time.perf_counter(), 0
    with (args.out_dir / "sr-log.jsonl").open("a") as log:
        for video, windows in todo.items():
            dst = args.out_dir / f"{video}.npy"
            if dst.exists():
                continue
            t = time.perf_counter()
            n = sr_video(model, args.frames_dir / f"{video}.npy", dst, windows, device, half)
            seconds = time.perf_counter() - t
            n_all += n
            log.write(json.dumps({"video": video, "frames": n, "seconds": seconds}) + "\n")
            log.flush()
            print(f"{video}: {n} frames, {seconds / n * 1000:.1f} ms/frame", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    total = time.perf_counter() - t_all
    print(f"done: {n_all} frames in {total:.0f} s ({total / max(n_all, 1) * 1000:.1f} ms/frame); "
          f"peak GPU {peak:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
