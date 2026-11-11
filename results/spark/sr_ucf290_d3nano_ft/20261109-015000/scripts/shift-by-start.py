#!/usr/bin/env python3
"""Score shift draft -> B, split by the draft score (Eric, 2026-11-09): do large events rise less than the
non-event windows of anomaly videos only because they start higher? CPU only.
Columns: mean shift in probability and in logit, overall and inside draft-score bins.

  uv run --frozen python <run>/scripts/shift-by-start.py --pair NAME:DRAFT:COND [...] --manifest M --events E
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location("layout_eval", REPO / "scripts/layout-eval.py")
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def groups(rows, manifest, events) -> dict[str, np.ndarray]:
    """Boolean masks: non-event windows of anomaly videos, and the event windows of each tier."""
    label = np.array([r["label"] for r in rows])
    in_normal_video = np.array([not manifest[r["video"]]["events"] for r in rows])
    out = {"anomaly-video negatives": (label == 0) & ~in_normal_video}
    for tier, (include, y) in le.tier_masks(rows, manifest, events).items():
        if tier in ("small", "large"):
            out[f"{tier} events"] = include & (y == 1)
    return out


def report(name: str, d: np.ndarray, c: np.ndarray, masks: dict[str, np.ndarray]) -> None:
    print(f"== {name}")
    print(f"{'group':26s} {'n':>5s} {'dp':>7s} {'dlogit':>7s}  " + "  ".join(f"[{a:.1f},{b:.1f}) dp n" for a, b in zip(BINS, BINS[1:])))
    for g, m in masks.items():
        cells = []
        for a, b in zip(BINS, BINS[1:]):
            inside = m & (d >= a) & (d < b if b < 1 else d <= b)
            cells.append(f"{(c[inside] - d[inside]).mean():+.3f} {inside.sum():4d}" if inside.sum() >= 20 else f"{'-':>6s} {inside.sum():4d}")
        print(f"{g:26s} {m.sum():5d} {(c[m] - d[m]).mean():+.3f} {(logit(c[m]) - logit(d[m])).mean():+.3f}  " + "  ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    args = ap.parse_args()
    manifest = {m["name"]: m for m in json.loads(args.manifest.read_text())}
    events = [json.loads(line) for line in args.events.read_text().splitlines()]
    for pair in args.pair:
        name, draft_path, cond_path = pair.split(":")
        draft, cond = le.load_scores(REPO / draft_path), le.load_scores(REPO / cond_path)
        keys = sorted(draft)
        rows = [draft[k] for k in keys]
        d = np.array([draft[k]["score"] for k in keys])
        c = np.array([cond[k]["score"] for k in keys])
        report(name, d, c, groups(rows, manifest, events))


if __name__ == "__main__":
    main()
