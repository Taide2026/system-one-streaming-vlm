#!/usr/bin/env python3
"""Zoom pilot on the case-3 split events (2026-10-30): does spatial cropping move p where every temporal fix failed?

  uv run --frozen python scripts/zoom-pilot.py --manifest <eval run's manifest.json> --root <ucf-crime asset> \\
      [--adapter <run dir>] --out <json>

Per event: 8 frames spread over the event span (the fine-tuned oracle timing: every frame inside the event), scored
(a) full frame — the oracle re-run; (b) TGM crop: the highest-motion region of a frame-difference heatmap over the
span (MoG-VLM's temporal gradient magnitude), a box of 1/2 and 1/3 of the frame, cropped on every frame and up-scaled
2x; (c) a center crop of 1/2 as a dumb-crop control. The V1 question, d3 backbone, one pass each. Also runs the TGM 1/2
crop on two windows of each normal pilot video (false-alarm control). Prints a table and writes --out."""
from __future__ import annotations

import argparse
import json
import sys
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

EVENTS = {"RoadAccidents010_x264.mp4": (7.7, 9.03), "RoadAccidents004_x264.mp4": (4.7, 6.33),
          "RoadAccidents011_x264.mp4": (8.7, 10.03), "RoadAccidents122_x264.mp4": (10.0, 12.03),
          "Shoplifting037_x264.mp4": (38.0, 40.03)}
NORMALS = ["Normal_Videos_345_x264.mp4", "Normal_Videos_881_x264.mp4", "Normal_Videos_878_x264.mp4",
           "Normal_Videos_745_x264.mp4", "Normal_Videos_889_x264.mp4"]


def frames_at(path: Path, times: list[float], fps: float) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, fr = cap.read()
            if ok:
                out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return out


def tgm_box(path: Path, t0: float, t1: float, fps: float, frac: float) -> tuple[int, int, int, int]:
    """Peak of the summed absolute frame difference over 16 frames of [t0, t1]: a frac-sized box around it."""
    fr = frames_at(path, [t0 + (k + 0.5) * (t1 - t0) / 16 for k in range(16)], fps)
    g = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32) for f in fr]
    heat = np.sum([np.abs(a - b) for a, b in zip(g[1:], g[:-1])], axis=0)
    heat = cv2.GaussianBlur(heat, (31, 31), 0)
    y, x = np.unravel_index(int(heat.argmax()), heat.shape)
    H, W = heat.shape
    h, w = int(H * frac), int(W * frac)
    y0, x0 = min(max(y - h // 2, 0), H - h), min(max(x - w // 2, 0), W - w)
    return x0, y0, w, h


def crop_images(frames: list[np.ndarray], box: tuple[int, int, int, int] | None) -> list[Image.Image]:
    out = []
    for f in frames:
        if box is not None:
            x0, y0, w, h = box
            f = f[y0:y0 + h, x0:x0 + w]
            f = cv2.resize(f, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        out.append(Image.fromarray(f))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--asset", default="d3-nano")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    man = {v["name"]: v for v in json.loads(args.manifest.read_text())}
    p = PROMPTS["v1"]
    question = Question(state=p["state"], text=p["text"], options=tuple(p["options"]), positive=p["positive"])
    spec = {"name": "d3", "asset": args.asset, "path": str(resolve(args.asset)), "memory_cap_gb": 40}
    if args.adapter:
        spec["adapter"] = args.adapter
    backbone = BACKBONES.build(spec)
    import torch

    def score(images) -> float:
        with torch.inference_mode():
            out = backbone.forward(ModelInput(images=images, text=None), question)
        z = np.asarray(out.logits, dtype=np.float32).flatten()[:2]
        e = np.exp(z - z.max())
        return float(e[0] / e.sum())

    results = {}
    for name, (t0, t1) in EVENTS.items():
        m = man[name]; fps = m["fps"]; path = args.root / m["path"]
        times = [t0 + (k + 0.5) * (t1 - t0) / 8 for k in range(8)]
        fr = frames_at(path, times, fps)
        row = {"oracle full": score(crop_images(fr, None))}
        for frac, label in ((0.5, "TGM crop 1/2"), (1 / 3, "TGM crop 1/3")):
            row[label] = score(crop_images(fr, tgm_box(path, t0, t1, fps, frac)))
        H, W = fr[0].shape[:2]
        row["center crop 1/2"] = score(crop_images(fr, (W // 4, H // 4, W // 2, H // 2)))
        results[name] = row
        print(name[:-9], {k: round(v, 2) for k, v in row.items()}, flush=True)
    for name in NORMALS:
        m = man[name]; fps = m["fps"]; path = args.root / m["path"]
        for w0 in (2.0, 6.0):
            t0, t1 = w0, w0 + 2.0
            fr = frames_at(path, [t0 + (k + 0.5) * 0.25 for k in range(8)], fps)
            if not fr:
                continue
            key = f"{name} @{w0:.0f}s"
            results[key] = {"full": score(crop_images(fr, None)),
                            "TGM crop 1/2": score(crop_images(fr, tgm_box(path, t0, t1, fps, 0.5)))}
            print(key, {k: round(v, 2) for k, v in results[key].items()}, flush=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
