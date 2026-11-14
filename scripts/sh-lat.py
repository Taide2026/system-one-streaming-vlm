#!/usr/bin/env python3
"""Shieldstral latency by stage (2026-11-13), the third model size (3.85B) of the where-to-save comparison.

The first two windows of each test video (5 warm-up windows left out). Rows: whole frame 1x, 1.5x, 2x; whole frame 2x
with two thirds of the visual tokens removed AFTER the vision tower (dropped at random; only the language model's time
is taken from that run, accuracy is not evaluated here); key frames + motion crops as plain images (input form keycrop).
Stages: decode, pre (enlarge / crop), proc (processor), vision (get_image_features), lm (language model on the embedded
sequence). The model has one attention implementation, so there is no fast / reference split.

  python scripts/sh-lat.py --config <eval yaml> --out <json> [--per-video 2]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import s1stream.plugins  # noqa: F401
from s1stream.config import load_config
from s1stream.input_forms.frames import read_frames
from s1stream.input_forms.keycrop import keycrop_images
from s1stream.input_forms.upsample import upsample
from s1stream.registry import BACKBONES, DATASETS
from s1stream.windows import plan_windows

ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--per-video", type=int, default=2); ap.add_argument("--warmup", type=int, default=5); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
config = load_config(a.config); bb = BACKBONES.build(config.backbone); q = config.question
model, tok = bb.model, bb.processor.tokenizer
IMG = tok.convert_tokens_to_ids("[IMG]")
items = list(DATASETS.build(config.dataset).items())
plan = [(it, w) for it in items for w in plan_windows(it, config.window.length_s, config.window.stride_s, config.window.frames)[:a.per_video]]
if a.limit:
    plan = plan[:a.limit]
rng = torch.Generator(device="cpu").manual_seed(0)


def sync():
    torch.cuda.synchronize(); return time.perf_counter()


def lm_only(embeds):
    lm = model.model.language_model if hasattr(model, "model") and hasattr(model.model, "language_model") else model.language_model
    out = lm(inputs_embeds=embeds)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return model.lm_head(h[:, -1:]) if hasattr(model, "lm_head") else h


def one(it, w, kind):
    t0 = sync(); images = read_frames(it.path, w.frame_indices); t1 = sync()
    if kind == "key":
        send = keycrop_images(images, (0, 7), 2.0, 32, 0.05)
    else:
        send = upsample(images, 2.0 if kind == "cut" else float(kind), "bicubic")
    t2 = sync(); inputs = bb.prepare(send, q); t3 = sync()
    with torch.inference_mode():
        feats = model.get_image_features(pixel_values=inputs["pixel_values"], image_sizes=inputs.get("image_sizes")); t4 = sync()
        ids = inputs["input_ids"]; emb = model.get_input_embeddings()(ids)
        if kind == "cut":                                           # keep a third of the image positions, at random
            pos = (ids[0] == IMG).nonzero()[:, 0]
            drop = pos[torch.randperm(len(pos), generator=rng)[: (2 * len(pos)) // 3].to(pos.device)]
            keepm = torch.ones(ids.shape[1], dtype=torch.bool, device=ids.device); keepm[drop] = False
            emb = emb[:, keepm]
        t5 = sync(); lm_only(emb); t6 = sync()
    n_img = int((ids[0] == IMG).sum()); n_img = n_img - (2 * n_img) // 3 if kind == "cut" else n_img
    return {"decode": t1 - t0, "pre": t2 - t1, "proc": t3 - t2, "vision": t4 - t3, "lm": t6 - t5, "total": (t4 - t0) + (t6 - t5), "tokens": n_img}


res = {"gpu": torch.cuda.get_device_name(), "windows": len(plan) - a.warmup, "rows": {}}
stages = ("decode", "pre", "proc", "vision", "lm", "total")
for name, kind in (("1x", "1.0"), ("1.5x", "1.5"), ("2x", "2.0"), ("2x, LM tokens cut to a third", "cut"), ("key frames + crops", "key")):
    rows = [one(it, w, kind) for it, w in plan][a.warmup:]
    r = {s: {"p50": 1000 * float(np.percentile([x[s] for x in rows], 50)), "mean": 1000 * float(np.mean([x[s] for x in rows]))} for s in stages}
    r["tokens"] = float(np.mean([x["tokens"] for x in rows])); res["rows"][name] = r
    print(f"shieldstral {name:30s} tokens {r['tokens']:6.0f} | total p50 {r['total']['p50']:5.0f} ms | mean ms: " + "  ".join(f"{s} {r[s]['mean']:.0f}" for s in stages[:-1]), flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
print("SHLAT_DONE")
