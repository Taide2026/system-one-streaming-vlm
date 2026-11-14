#!/usr/bin/env python3
"""Shieldstral smoke test and stage timing (2026-11-13): python scripts/sh-smoke.py <eval yaml> [n_windows]
Prints the rendered prompt once, then per input (1x, 1.5x, 2x whole frame) the visual tokens, the score on a few
positive and negative windows of the first anomalous videos, and mean time per stage."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

import s1stream.plugins  # noqa: F401  registers every plugin
from s1stream.config import load_config
from s1stream.input_forms.frames import read_frames
from s1stream.input_forms.upsample import upsample
from s1stream.registry import BACKBONES, DATASETS
from s1stream.windows import plan_windows

config = load_config(Path(sys.argv[1])); n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
bb = BACKBONES.build(config.backbone)
q = config.question
items = [it for it in DATASETS.build(config.dataset).items() if it.events][:6]
plan = []
for it in items:
    ws = plan_windows(it, config.window.length_s, config.window.stride_s, config.window.frames)
    pos = [w for w in ws if w.label == 1][:n // 12]; neg = [w for w in ws if w.label == 0][:n // 12]
    plan += [(it, w) for w in pos + neg]
print("windows", len(plan), "positives", sum(w.label for _, w in plan), flush=True)


def sync():
    torch.cuda.synchronize(); return time.perf_counter()


first = True
for factor in (1.0, 1.5, 2.0):
    rows = []
    for it, w in plan:
        t0 = sync(); images = read_frames(it.path, w.frame_indices); t1 = sync()
        images = upsample(images, factor, "bicubic"); t2 = sync()
        inputs = bb.prepare(images, q); t3 = sync()
        if first:
            ids = inputs["input_ids"][0]; tok = bb.processor.tokenizer
            img_id = tok.convert_tokens_to_ids("[IMG]")
            txt = tok.decode(ids[ids != img_id]); print("PROMPT (image tokens removed):", repr(txt[:900]), flush=True)
            print("input keys", {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in inputs.items()}, "yes ids", bb.ids["yes"][:6], "no ids", bb.ids["no"][:6], flush=True)
            first = False
        with torch.inference_mode():
            feats = bb.model.get_image_features(pixel_values=inputs["pixel_values"], image_sizes=inputs.get("image_sizes"))
            t4 = sync()
            logits = bb.model(**inputs).logits[0, -1]
            t5 = sync()
        z = bb.read(logits, q); p = float(torch.softmax(z, 0)[0])
        n_img = int((inputs["input_ids"][0] == bb.processor.tokenizer.convert_tokens_to_ids("[IMG]")).sum())
        rows.append((w.label, p, n_img, int(inputs["input_ids"].shape[1]), t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4))
    r = np.array(rows[3:], dtype=float)
    pos, neg = r[r[:, 0] == 1, 1], r[r[:, 0] == 0, 1]
    auc = float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()) if len(pos) and len(neg) else float("nan")
    print(f"factor {factor}: image tokens {r[:, 2].mean():.0f} sequence {r[:, 3].mean():.0f} | P(yes) positives {pos.mean():.3f} negatives {neg.mean():.3f} AUC on these windows {100 * auc:.1f} | "
          f"ms: decode {1000 * r[:, 4].mean():.0f} enlarge {1000 * r[:, 5].mean():.0f} processor {1000 * r[:, 6].mean():.0f} vision alone {1000 * r[:, 7].mean():.0f} full forward {1000 * r[:, 8].mean():.0f}", flush=True)
print("SH_SMOKE_DONE")
