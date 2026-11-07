#!/usr/bin/env python3
"""Score shift draft -> B (2026-11-06): do normal windows also score higher after the 2x up-scale?
Negatives = every label-0 window. Positives per tier use the tier rule of scripts/layout-eval.py. CPU only.

  uv run --frozen python <run>/scripts/score-shift.py --draft <scores.jsonl> --cond <scores.jsonl> \
      --manifest <manifest.json> --events <jev-hard-events.jsonl>
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("layout_eval", Path(__file__).resolve().parents[5] / "scripts/layout-eval.py")
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)


def describe(name: str, d: np.ndarray, c: np.ndarray) -> None:
    """One table row: mean, median, top-1 % quantile, share above 0.5, for draft and the condition."""
    print(f"{name:26s} n={len(d):6d}  mean {d.mean():.3f} -> {c.mean():.3f}  median {np.median(d):.3f} -> {np.median(c):.3f}  "
          f"p99 {np.quantile(d, .99):.3f} -> {np.quantile(c, .99):.3f}  >0.5 {np.mean(d > .5) * 100:.1f}% -> {np.mean(c > .5) * 100:.1f}%  "
          f"higher {np.mean(c > d) * 100:.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("--draft", "--cond", "--manifest", "--events"):
        ap.add_argument(a, type=Path, required=True)
    args = ap.parse_args()
    manifest = {m["name"]: m for m in json.loads(args.manifest.read_text())}
    events = [json.loads(line) for line in args.events.read_text().splitlines()]
    draft, cond = le.load_scores(args.draft), le.load_scores(args.cond)
    keys = sorted(draft)
    rows = [draft[k] for k in keys]
    d = np.array([draft[k]["score"] for k in keys])
    c = np.array([cond[k]["score"] for k in keys])
    label = np.array([r["label"] for r in rows])
    in_normal_video = np.array([not manifest[r["video"]]["events"] for r in rows])

    describe("negatives, all", d[label == 0], c[label == 0])
    describe("negatives, normal videos", d[(label == 0) & in_normal_video], c[(label == 0) & in_normal_video])
    describe("negatives, anomaly videos", d[(label == 0) & ~in_normal_video], c[(label == 0) & ~in_normal_video])
    masks = le.tier_masks(rows, manifest, events)
    for t in ("small", "medium", "large"):
        include, y = masks[t]
        pos = include & (y == 1)
        describe(f"positives, {t}", d[pos], c[pos])


if __name__ == "__main__":
    main()
