#!/usr/bin/env python3
"""Score training records with a backbone (diagnostics for the fine-tune, 2026-10-24).

  PYTHONPATH=src python scripts/score-records.py <records.jsonl> <train config.yaml> [--adapter <run dir>]
                                                 [--limit N] [--out scores.jsonl]

Builds the backbone of the training config (clef_flash or d3), with the adapter when given, runs every record's
frames through it with the config's question, and prints the window AUC of p(positive) against y >= 0.5, the soft
cross-entropy against (y, 1 - y), the mean p on records with y == 0 and y == 1, and the fraction of records with
p >= 0.5. Zero-shot (no adapter) against fine-tuned on the same held-out records tells whether the fine-tune moved the
model at all, and in which direction, before the 290-video evaluation finishes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import s1stream.plugins  # noqa: F401
from s1stream.assets import resolve
from s1stream.input_forms.frames import read_frames
from s1stream.metrics import auc
from s1stream.registry import ANSWER_HEADS, BACKBONES
from s1stream.train import load_train_config, question_of
from s1stream.traindata import read_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data = load_train_config(args.config)
    question = question_of(data)
    spec = dict(data["backbone"])
    spec["path"] = str(resolve(spec["asset"]))
    if args.adapter:
        spec["adapter"] = str(args.adapter)
    backbone = BACKBONES.build(spec)
    head = ANSWER_HEADS.build({"name": "option_logits"})
    records = read_records(args.records)[:args.limit]
    positive = question.positive if isinstance(question.positive, str) else question.positive[0]
    ps, ys = [], []
    out = args.out.open("w") if args.out else None
    for i, record in enumerate(records):
        images = read_frames(Path(record.path), record.frame_indices)
        p = head.probabilities(backbone.forward(__import__("s1stream.types", fromlist=["ModelInput"]).ModelInput(images=images), question), question)[positive]
        ps.append(p)
        ys.append(record.y)
        if out:
            out.write(json.dumps({"video": record.video, "window": record.window, "y": record.y, "p": p}) + "\n")
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(records)}", flush=True)
    p, y = np.array(ps), np.array(ys)
    eps = 1e-6
    nll = float(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean())
    result = {"records": len(records), "adapter": str(args.adapter) if args.adapter else None,
              "window_auc": auc(p.tolist(), (y >= 0.5).astype(int).tolist()), "soft_nll": nll,
              "mean_p_y0": float(p[y == 0].mean()) if np.any(y == 0) else None,
              "mean_p_y1": float(p[y == 1].mean()) if np.any(y == 1) else None,
              "mean_p_partial": float(p[(y > 0) & (y < 1)].mean()) if np.any((y > 0) & (y < 1)) else None,
              "frac_p_ge_0.5": float((p >= 0.5).mean())}
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
