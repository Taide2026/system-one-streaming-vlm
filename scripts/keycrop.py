#!/usr/bin/env python3
"""Keyframe + motion crops, the cost version (2026-11-13): the vision tower only sees what is kept.

keyframe.py measured the effect of "key frames in full, the other frames only on the moving region" by running every
frame through the vision tower at 2x and dropping tokens afterwards. Here the saving is real:
  key frames    enlarged 2x in full and encoded (300 tokens each)
  other frames  only the bounding box of the moving blocks is cut out of the ORIGINAL frame, enlarged 2x and encoded
                as a small image of its own; its tokens are placed at the positions that region has in the 2x layout
Nothing else is pre-processed or encoded. The prompt layout (token ids, positions) of an 8-frame 2x window is the
same for every window of a dataset with one frame size, so it is built once and reused.

  python scripts/keycrop.py eval    --config <eval yaml> --out <run dir> --key 0 7 --rest touch
  python scripts/keycrop.py latency --config <eval yaml> --out <json> --kernels ref|fla   (all layouts, stage by stage)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["eval", "latency"])
ap.add_argument("--config", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--key", type=int, nargs="*", default=[0, 7]); ap.add_argument("--rest", default="touch", choices=["touch", "loose"])
ap.add_argument("--key-scale", type=int, default=2, choices=[1, 2], help="1: key frames at the original resolution (one token per 2x2 block)")
ap.add_argument("--crop-all", type=int, default=0, help="1: the motion crop is also taken from the key frames and laid over them")
ap.add_argument("--kernels", default="ref", choices=["ref", "fla"]); ap.add_argument("--per-video", type=int, default=2)
ap.add_argument("--warmup", type=int, default=5); ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
if args.cmd == "latency" and args.kernels == "ref":
    sys.modules["fla"] = None

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech import Core, tgm_mask, upsample  # noqa: E402

from s1stream.config import load_config  # noqa: E402
from s1stream.input_forms.frames import read_frames  # noqa: E402
from s1stream.registry import BACKBONES, DATASETS  # noqa: E402
from s1stream.windows import plan_windows  # noqa: E402

config = load_config(args.config)
backbone = BACKBONES.build(config.backbone); core = Core(backbone, config.question)
ip = core.rt.processor.image_processor
ip.size = {"shortest_edge": 1024, "longest_edge": ip.size["longest_edge"]}     # let a small crop keep its size
items = list(DATASETS.build(config.dataset).items())
LAYOUT = {}      # frame size -> the cached 2x prompt inputs


def sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def layout_for(images: list) -> dict:
    size = images[0].size
    if size not in LAYOUT:
        LAYOUT[size] = core.encode(upsample(images, 2.0, "bicubic"))
    return LAYOUT[size]


def one(images: list, key: list, rest: str, timed: bool = False, key_scale: int = 2, crop_all: int = 0) -> dict:
    """One window through the keyframe + crop path. Returns margin, token and patch counts, stage times."""
    big = layout_for(images)                                    # cached after the first window: not part of the cost
    g2 = big["grids"][0]; hb, wb = math.ceil(g2[0] / 2), math.ceil(g2[1] / 2)
    width, height = images[0].size
    t0 = sync() if timed else 0.0
    fine = cv2.resize(tgm_mask(images), (wb, hb), interpolation=cv2.INTER_AREA) > 0.05
    if rest == "loose":
        fine = cv2.dilate(fine.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    others = list(range(len(images))) if crop_all else [k for k in range(len(images)) if k not in key]
    if key_scale == 2:
        send = [Image.fromarray(cv2.resize(np.asarray(images[k].convert("RGB")), (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)) for k in key]
    else:
        send = [images[k].convert("RGB") for k in key]
    rect = None
    if fine.any() and others:
        rr, cc = np.where(fine); r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
        x0, x1, y0, y1 = c0 * 32, min((c1 + 1) * 32, width), r0 * 32, min((r1 + 1) * 32, height)
        rect = (2 * r0, min(2 * (r1 + 1), g2[0]), 2 * c0, min(2 * (c1 + 1), g2[1]))          # token rows / cols in the 2x layout
        for k in others:
            a = np.asarray(images[k].convert("RGB"))[y0:y1, x0:x1]
            send.append(Image.fromarray(cv2.resize(a, ((x1 - x0) * 2, (y1 - y0) * 2), interpolation=cv2.INTER_CUBIC)))
    t1 = sync() if timed else 0.0
    enc = ip(images=send, return_tensors="pt")
    pixel_values, grid = enc["pixel_values"].to(core.rt.device), enc["image_grid_thw"].to(core.rt.device)
    t2 = sync() if timed else 0.0
    feats = list(core.model.get_image_features(pixel_values, grid, return_dict=True).pooler_output)
    t3 = sync() if timed else 0.0
    dim, dev, dt = feats[0].shape[-1], feats[0].device, feats[0].dtype
    toks = [torch.zeros(g2[0] * g2[1], dim, device=dev, dtype=dt) for _ in images]
    keep = [torch.zeros(g2, dtype=torch.bool, device=dev) for _ in images]
    for i, k in enumerate(key):
        if key_scale == 2:
            assert feats[i].shape[0] == g2[0] * g2[1], (feats[i].shape, g2)
            toks[k] = feats[i].clone(); keep[k][:] = True
        else:                                   # 1x tokens sit on the top-left token of each 2x2 block
            assert feats[i].shape[0] == hb * wb, (feats[i].shape, hb, wb)
            toks[k].view(g2[0], g2[1], dim)[0::2, 0::2] = feats[i].view(hb, wb, dim); keep[k][0::2, 0::2] = True
    if rect is not None:
        ra, rb, ca, cb = rect
        for i, k in enumerate(others):
            f = feats[len(key) + i]
            assert f.shape[0] == (rb - ra) * (cb - ca), (f.shape, rect)
            t = toks[k].view(g2[0], g2[1], dim); t[ra:rb, ca:cb] = f.view(rb - ra, cb - ca, dim); keep[k][ra:rb, ca:cb] = True
    t4 = sync() if timed else 0.0
    margin = core.language(big, toks, keep=keep)
    t5 = sync() if timed else 0.0
    return {"margin": margin, "tokens": int(sum(int(k.sum()) for k in keep)), "patches": int(grid.prod(-1).sum()),
            "pre": t1 - t0, "proc": t2 - t1, "vision": t3 - t2, "tokens_t": t4 - t3, "lm": t5 - t4}


if args.cmd == "eval":
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.yaml").write_text(yaml.safe_dump({**config.raw, "keycrop": {"key": args.key, "rest": args.rest, "key_scale": args.key_scale, "crop_all": args.crop_all}}, sort_keys=False, default_flow_style=None))
    n = 0
    with (args.out / "scores.jsonl").open("w") as stream, torch.inference_mode(), backbone.code.sdpa_backends(core.rt.device):
        for item in items:
            for window in plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames):
                r = one(read_frames(item.path, window.frame_indices), args.key, args.rest, key_scale=args.key_scale, crop_all=args.crop_all)
                stream.write(json.dumps({"video": item.name, "window": window.index, "start_s": window.start_s, "end_s": window.end_s,
                                         "label": window.label, "score": 1 / (1 + math.exp(-r["margin"])), "margin": r["margin"],
                                         "tokens": r["tokens"], "patches": r["patches"]}) + "\n")
                n += 1
                if n % 2000 == 0:
                    stream.flush(); print(f"{n} windows", flush=True)
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break
    (args.out / "run.json").write_text(json.dumps({"windows": n, "finished": datetime.now(timezone.utc).isoformat()}))
    print("KEYCROP_DONE", n, flush=True)
else:
    plan = [(item, w) for item in items for w in plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames)[:args.per_video]]
    if args.limit:
        plan = plan[:args.limit]
    try:
        import fla  # noqa: F401
        kern = "fla"
    except Exception:
        kern = "reference"
    result = {"gpu": torch.cuda.get_device_name(), "kernels": kern, "windows": len(plan) - args.warmup, "rows": {}}
    stages = ("decode", "pre", "proc", "vision", "tokens_t", "lm", "total")
    with torch.inference_mode(), backbone.code.sdpa_backends(core.rt.device):
        ALL = list(range(8))
        for name, key, rest, ks, ca in (("key3 + touch", [3], "touch", 2, 0), ("key3 + loose", [3], "loose", 2, 0), ("key0,7 + touch", [0, 7], "touch", 2, 0), ("key0,3,7 + touch", [0, 3, 7], "touch", 2, 0),
                                        ("key0,7 + loose", [0, 7], "loose", 2, 0), ("1x key0,7 + touch", [0, 7], "touch", 1, 0),
                                        ("1x key0,7 + loose", [0, 7], "loose", 1, 0), ("1x all + touch", ALL, "touch", 1, 1), ("1x all + loose", ALL, "loose", 1, 1)):
            rows = []
            for item, w in plan:
                t0 = sync(); images = read_frames(item.path, w.frame_indices); t1 = sync()
                r = one(images, key, rest, timed=True, key_scale=ks, crop_all=ca); r["decode"] = t1 - t0
                r["total"] = r["decode"] + r["pre"] + r["proc"] + r["vision"] + r["tokens_t"] + r["lm"]; rows.append(r)
            rows = rows[args.warmup:]
            out = {s: {"p50": 1000 * float(np.percentile([x[s] for x in rows], 50)), "p95": 1000 * float(np.percentile([x[s] for x in rows], 95)),
                       "mean": 1000 * float(np.mean([x[s] for x in rows]))} for s in stages}
            out["n_tokens"] = float(np.mean([x["tokens"] for x in rows])); out["n_patches"] = float(np.mean([x["patches"] for x in rows]))
            result["rows"][name] = out
            print(f"{kern:9s} {name:18s} tokens {out['n_tokens']:6.0f} patches {out['n_patches']:6.0f} | total p50 {out['total']['p50']:5.0f} p95 {out['total']['p95']:5.0f} ms | mean ms: "
                  + "  ".join(f"{s} {out[s]['mean']:.0f}" for s in stages[:-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print("KEYCROP_LAT_DONE", flush=True)
