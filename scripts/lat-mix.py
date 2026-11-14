#!/usr/bin/env python3
"""Latency of the mixed-resolution token layouts, stage by stage (2026-11-12). One GPU, nothing else on it.

The first 2 windows of each test video (5 warm-up windows left out), each timed through the same taken-apart path:
  decode   read the 8 frames
  pre      bicubic upsampling, and for the mixed layouts the motion map
  proc     prompt + processor (resize, patchify, to device)
  vision   vision tower
  tokens   pooling / selecting the visual tokens (mixed layouts only)
  lm       language model + read-out
Inputs: 1x, 1.5x, 2x (tokens untouched) and, on the 2x frame, all coarse / tight / touch / loose.

  python scripts/lat-mix.py --config <eval yaml> --out <json> [--kernels ref|fla] [--per-video 2]
--kernels ref hides flash-linear-attention, so the reference kernels run (the numbers latency claims may use).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--config", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--kernels", default="ref", choices=["ref", "fla"]); ap.add_argument("--per-video", type=int, default=2)
ap.add_argument("--warmup", type=int, default=5); ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
if args.kernels == "ref":
    sys.modules["fla"] = None

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech import Core, tgm_mask, upsample  # noqa: E402

from s1stream.config import load_config  # noqa: E402
from s1stream.input_forms.frames import read_frames  # noqa: E402
from s1stream.registry import BACKBONES, DATASETS  # noqa: E402
from s1stream.windows import plan_windows  # noqa: E402

config = load_config(args.config)
backbone = BACKBONES.build(config.backbone); core = Core(backbone, config.question)
items = list(DATASETS.build(config.dataset).items())
plan = [(item, w) for item in items for w in plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames)[:args.per_video]]
if args.limit:
    plan = plan[:args.limit]
try:
    import fla  # noqa: F401
    have_fla = True
except Exception:
    have_fla = False

CONFIGS = [("1x", 1.0, None), ("1.25x", 1.25, None), ("1.5x", 1.5, None), ("1.75x", 1.75, None), ("2x", 2.0, None),
           ("2x all coarse", 2.0, "coarse"), ("2x tight", 2.0, "tight"), ("2x touch", 2.0, "touch"), ("2x loose", 2.0, "loose")]


def sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def one(item, window, factor: float, layout: str | None) -> dict:
    t0 = sync()
    images = read_frames(item.path, window.frame_indices)
    t1 = sync()
    big_images = upsample(images, factor, "bicubic")
    cov = None
    if layout in ("tight", "touch", "loose"):
        mask = tgm_mask(images)
    t2 = sync()
    inputs = core.encode(big_images)
    t3 = sync()
    feats = core.vision(inputs)
    t4 = sync()
    keep, toks = None, feats
    if layout is not None:
        g2 = inputs["grids"][0]; hb, wb = math.ceil(g2[0] / 2), math.ceil(g2[1] / 2)
        if layout == "coarse":
            fine = np.zeros((hb, wb), bool)
        else:
            cov = cv2.resize(mask, (wb, hb), interpolation=cv2.INTER_AREA)
            fine = cov > (0.25 if layout == "tight" else 0.05)
            if layout == "loose":
                fine = cv2.dilate(fine.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        up = lambda t: t.repeat_interleave(2, 0).repeat_interleave(2, 1)[:g2[0], :g2[1]]
        ft = up(torch.as_tensor(fine, device=feats[0].device))
        anchor = torch.zeros(g2, dtype=torch.bool, device=feats[0].device); anchor[0::2, 0::2] = True
        toks, keep = [], []
        for f in feats:
            x = f.view(g2[0], g2[1], -1)
            c = torch.nn.functional.avg_pool2d(x.permute(2, 0, 1)[None].float(), 2, 2, ceil_mode=True, count_include_pad=False)[0]
            rep = up(c.permute(1, 2, 0)).to(x.dtype)
            toks.append(torch.where(ft[..., None], x, rep).reshape(g2[0] * g2[1], -1)); keep.append(ft | anchor)
    t5 = sync()
    core.language(inputs, toks, keep=keep)
    t6 = sync()
    n_tok = int(sum(int(k.sum()) for k in keep)) if keep is not None else int(sum(f.shape[0] for f in feats))
    return {"decode": t1 - t0, "pre": t2 - t1, "proc": t3 - t2, "vision": t4 - t3, "tokens": t5 - t4, "lm": t6 - t5, "total": t6 - t0, "n_tokens": n_tok}


result = {"gpu": torch.cuda.get_device_name(), "kernels": "fla" if have_fla else "reference", "windows": len(plan) - args.warmup,
          "config": str(args.config), "rows": {}}
stages = ("decode", "pre", "proc", "vision", "tokens", "lm", "total")
with torch.inference_mode(), backbone.code.sdpa_backends(core.rt.device):
    for name, factor, layout in CONFIGS:
        rows = [one(item, w, factor, layout) for item, w in plan][args.warmup:]
        r = {s: {"p50": 1000 * float(np.percentile([x[s] for x in rows], 50)), "p95": 1000 * float(np.percentile([x[s] for x in rows], 95)),
                 "mean": 1000 * float(np.mean([x[s] for x in rows]))} for s in stages}
        r["n_tokens"] = float(np.mean([x["n_tokens"] for x in rows]))
        result["rows"][name] = r
        print(f"{result['kernels']:9s} {name:14s} tokens {r['n_tokens']:6.0f} | total p50 {r['total']['p50']:5.0f} p95 {r['total']['p95']:5.0f} ms | mean ms: "
              + "  ".join(f"{s} {r[s]['mean']:.0f}" for s in stages[:-1]), flush=True)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=1))
print("LAT_DONE", args.out, flush=True)
