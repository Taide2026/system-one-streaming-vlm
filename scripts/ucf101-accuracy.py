#!/usr/bin/env python3
"""Top-1 / top-5 accuracy of a UCF101 run (2026-10-28): scripts/ucf101-accuracy.py <run dir> [<run dir> ...]

A clip's class probability vector is the mean of its windows' option probabilities (scores.jsonl); the label is the
manifest's category. Prints per run: clips, windows, top-1, top-5, and a Typst row; --per-class adds the worst classes.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--per-class", type=int, default=0, help="print the N worst classes")
    args = ap.parse_args()
    for run in args.runs:
        manifest = {v["name"]: v for v in json.loads((run / "manifest.json").read_text())}
        rows = [json.loads(l) for l in (run / "scores.jsonl").read_text().splitlines() if l.strip()]
        options = list(rows[0]["probabilities"])
        by_clip: dict[str, list[np.ndarray]] = defaultdict(list)
        for r in rows:
            by_clip[r["video"]].append(np.array([r["probabilities"][o] for o in options]))
        top1, top5, per = 0, 0, defaultdict(lambda: [0, 0])
        for clip, vecs in by_clip.items():
            p = np.mean(vecs, axis=0)
            truth = manifest[clip]["category"]
            order = [options[i] for i in np.argsort(-p)]
            per[truth][1] += 1
            if order[0].lower() == truth.lower():
                top1 += 1; per[truth][0] += 1
            if truth.lower() in [o.lower() for o in order[:5]]:
                top5 += 1
        n = len(by_clip)
        print(f"{run}: {n} clips ({len(rows)} windows, {len(manifest) - n} clips without a window), "
              f"top-1 {100 * top1 / n:.1f} %, top-5 {100 * top5 / n:.1f} %")
        print(f"  // Typst: [{run.parent.name}], [{100 * top1 / n:.1f}], [{100 * top5 / n:.1f}],")
        if args.per_class:
            worst = sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1])[:args.per_class]
            print("  worst: " + ", ".join(f"{c} {100 * k / t:.0f} % ({t})" for c, (k, t) in worst))


if __name__ == "__main__":
    main()
