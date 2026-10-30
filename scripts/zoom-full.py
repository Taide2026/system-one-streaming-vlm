#!/usr/bin/env python3
"""Zoom second pass over every window of an eval run (2026-10-30), so any trigger policy can be scored offline:
--frac picks the crop size (1/3 default, 1/2 for the wider box).

  uv run --frozen python scripts/zoom-full.py --run <eval run dir> --root <ucf-crime asset> --adapter <ft run> --out <jsonl>

Per scores.jsonl row: the window's 8 frames; TGM = summed absolute difference of those frames (blurred), its peak gives
a 1/3-of-frame box; the crop, up-scaled 2x, is scored with the same V1 question. Writes one line per window:
{video, window, label, p_draft, p_zoom, tgm} with tgm = the heat map's peak value (motion energy, for the trigger).
Resumable: appends after the last written window. Policies (max / replace, delta band, TGM threshold) are applied
offline by scripts/zoom-eval.py."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--asset", default="d3-nano")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frac", type=float, default=1 / 3, help="crop box side as a fraction of the frame (1/3 or 1/2)")
    ap.add_argument("--mode", choices=["tgm", "center"], default="tgm", help="box at the TGM peak, or fixed at the frame centre")
    args = ap.parse_args()
    man = {v["name"]: v for v in json.loads((args.run / "manifest.json").read_text())}
    rows = [json.loads(l) for l in (args.run / "scores.jsonl").read_text().splitlines() if l.strip()]
    done = set()
    if args.out.exists():
        done = {(r["video"], r["window"]) for r in map(json.loads, args.out.read_text().splitlines()) if r}
        print(f"resuming: {len(done)} windows already scored", flush=True)
    p = PROMPTS["v1"]
    question = Question(state=p["state"], text=p["text"], options=tuple(p["options"]), positive=p["positive"])
    spec = {"name": "d3", "asset": args.asset, "path": str(resolve(args.asset)), "memory_cap_gb": 40}
    if args.adapter:
        spec["adapter"] = args.adapter
    backbone = BACKBONES.build(spec)
    import time

    import torch
    t0, n = time.perf_counter(), 0
    caps: dict[str, cv2.VideoCapture] = {}
    with args.out.open("a") as f:
        for r in rows:
            key = (r["video"], r["window"])
            if key in done:
                continue
            m = man[r["video"]]
            if r["video"] not in caps:
                for c in caps.values():
                    c.release()
                caps = {r["video"]: cv2.VideoCapture(str(args.root / m["path"]))}
            cap = caps[r["video"]]
            fps = m["fps"]
            frames = []
            for k in range(8):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((r["start_s"] + (k + 0.5) * 0.25) * fps)))
                ok, fr = cap.read()
                if ok:
                    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if len(frames) < 4:
                continue
            g = [cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32) for x in frames]
            heat = cv2.GaussianBlur(np.sum([np.abs(a - b) for a, b in zip(g[1:], g[:-1])], axis=0), (31, 31), 0)
            H, W = heat.shape
            h, w = int(H * args.frac), int(W * args.frac)
            if args.mode == "center":
                y0, x0 = (H - h) // 2, (W - w) // 2
            else:
                y, x = np.unravel_index(int(heat.argmax()), heat.shape)
                y0, x0 = min(max(y - h // 2, 0), H - h), min(max(x - w // 2, 0), W - w)
            images = [Image.fromarray(cv2.resize(fr[y0:y0 + h, x0:x0 + w], (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))
                      for fr in frames]
            with torch.inference_mode():
                out = backbone.forward(ModelInput(images=images, text=None), question)
            z = np.asarray(out.logits, dtype=np.float32).flatten()[:2]
            e = np.exp(z - z.max())
            f.write(json.dumps({"video": r["video"], "window": r["window"], "label": r["label"],
                                "p_draft": r["score"], "p_zoom": float(e[0] / e.sum()), "tgm": float(heat.max())}) + "\n")
            n += 1
            if n % 500 == 0:
                f.flush()
                print(f"{n} windows, {(time.perf_counter() - t0) / n:.2f} s each", flush=True)
    print(f"done: {n} new windows -> {args.out}")


if __name__ == "__main__":
    main()
