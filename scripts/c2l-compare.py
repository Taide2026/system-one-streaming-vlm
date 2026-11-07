#!/usr/bin/env python3
"""Compare the TGM size proxy with UCFCrime2Local box areas (2026-11-05). No model - CPU.

  uv run --frozen python scripts/c2l-compare.py --boxes <dir>/c2l-events.jsonl --tgm <dir>/tgm-events.jsonl \\
      --t1 0.025152 --t2 0.044659 [--out <dir>/compare.json]

--tgm is the output of scripts/tgm-size.py on the manifest of scripts/c2l-events.py. --t1 / --t2 are the test-set
tier cuts (small < t1 <= medium < t2 <= large) from tgm-size.py on the 290 test videos.
Reports: Spearman rank correlation of TGM area vs box area; the 3 x 3 table of tiers (TGM tier by the test cuts,
box tier by the box-area terciles of these events, since box area and motion area have different scales); and the
share of TGM-small events whose box is in the box-small tercile.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TIERS = ("small", "medium", "large")


def tier_of(value: float, t1: float, t2: float) -> str:
    """small < t1 <= medium < t2 <= large."""
    if value < t1:
        return "small"
    if value < t2:
        return "medium"
    return "large"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties are not needed: areas are continuous)."""
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--tgm", type=Path, required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--t2", type=float, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    boxes = {(r["video"], tuple(r["event"])): r for r in map(json.loads, args.boxes.read_text().splitlines())}
    tgm = {(r["video"], tuple(r["event"])): r for r in map(json.loads, args.tgm.read_text().splitlines())}
    keys = sorted(boxes.keys() & tgm.keys())
    box_a = np.array([boxes[k]["box_area"] for k in keys])
    tgm_a = np.array([tgm[k]["area_frac"] for k in keys])
    b1, b2 = np.quantile(box_a, [1 / 3, 2 / 3])

    table = {t: {u: 0 for u in TIERS} for t in TIERS}  # table[tgm tier][box tier]
    for b, g in zip(box_a, tgm_a):
        table[tier_of(g, args.t1, args.t2)][tier_of(b, b1, b2)] += 1
    n_small = sum(table["small"].values())
    result = {
        "events": len(keys), "events_without_tgm": len(boxes) - len(keys),
        "spearman": spearman(tgm_a, box_a),
        "box_area_median": float(np.median(box_a)), "tgm_area_median": float(np.median(tgm_a)),
        "box_terciles": [float(b1), float(b2)], "test_cuts": [args.t1, args.t2],
        "table_tgm_by_box": table,
        "tgm_small_with_box_small": table["small"]["small"] / n_small if n_small else None,
        "agreement": sum(table[t][t] for t in TIERS) / len(keys),
    }
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
