#!/usr/bin/env python3
"""Per-tier frame AUC / AP of the layout test (2026-11-05), with paired bootstrap intervals against the draft.
No model - CPU.

  uv run --frozen python scripts/layout-eval.py --manifest <manifest.json> --events <jev-hard-events.jsonl> \\
      --draft <draft scores.jsonl> --run draft:<p> A:<p> B:<p> C:<p> [--gate 0.045] [--out <json>]

Tiers and the AUC / AP convention follow scripts/jev-hard.py: positives = windows inside the tier's events,
negatives = every negative window of the test set. --gate adds a "<name>+gate" row per layout: windows whose TGM
hot area >= gate take the draft score (fov2 skips the crop there). Bootstrap: 1,000 draws, seed 20261022,
anomaly and normal videos drawn separately; each draw scores the layout and the draft on the same videos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TIERS = ("small", "medium", "large", "all")
DRAWS, SEED = 1000, 20261022


def auc_ap(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Frame AUC and AP, the same formulas as scripts/jev-hard.py."""
    order = np.argsort(-s)
    y = y[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    tpr, fpr = tp / max(tp[-1], 1), fp / max(fp[-1], 1)
    auc = float(np.trapezoid(tpr, fpr))
    ap = float((tp / np.arange(1, len(y) + 1) * y).sum() / max(y.sum(), 1))
    return auc, ap


def load_scores(path: Path) -> dict[tuple[str, int], dict]:
    return {(r["video"], r["window"]): r for r in map(json.loads, path.read_text().splitlines())}


def tier_masks(rows: list[dict], manifest: dict, events: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per tier: (include, y) over rows. Same rule as jev-hard.py."""
    a = np.array([e["area_frac"] for e in events])
    t1, t2 = np.quantile(a, [1 / 3, 2 / 3])
    tiers: dict[str, list] = {}
    for e in events:
        t = "small" if e["area_frac"] < t1 else ("medium" if e["area_frac"] < t2 else "large")
        tiers.setdefault(e["video"], []).append((e["event"][0], e["event"][1], t))
    masks = {t: (np.zeros(len(rows), bool), np.zeros(len(rows))) for t in TIERS}
    for i, r in enumerate(rows):
        m = manifest[r["video"]]
        masks["all"][0][i], masks["all"][1][i] = True, r["label"]
        if not r["label"]:  # every negative window counts in every tier
            for t in TIERS[:3]:
                masks[t][0][i] = True
            continue
        w0, w1 = r["start_s"] * m["fps"], r["end_s"] * m["fps"]
        for e0, e1, t in tiers.get(r["video"], []):
            if w0 < e1 + 1 and w1 > e0:
                masks[t][0][i], masks[t][1][i] = True, 1
    return masks


def metrics(scores: np.ndarray, masks: dict, keep: np.ndarray) -> dict[str, tuple[float, float]]:
    """{tier: (AUC, AP)} over the rows where keep is True (keep may repeat rows: it is an index array)."""
    out = {}
    for t in TIERS:
        include, y = masks[t]
        idx = keep[include[keep]]
        out[t] = auc_ap(y[idx], scores[idx])
    return out


def bootstrap_index(rows: list[dict], manifest: dict, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one draw: anomaly and normal videos resampled separately, all windows of a drawn video."""
    by_video: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_video.setdefault(r["video"], []).append(i)
    anomaly = [v for v in by_video if manifest[v]["events"]]
    normal = [v for v in by_video if not manifest[v]["events"]]
    drawn = list(rng.choice(anomaly, len(anomaly))) + list(rng.choice(normal, len(normal)))
    return np.concatenate([by_video[v] for v in drawn])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--run", nargs="+", required=True, help="name:path to a scores.jsonl")
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    manifest = {m["name"]: m for m in json.loads(args.manifest.read_text())}
    events = [json.loads(line) for line in args.events.read_text().splitlines()]
    draft = load_scores(args.draft)
    keys = sorted(draft)
    rows = [draft[k] for k in keys]
    masks = tier_masks(rows, manifest, events)
    draft_s = np.array([r["score"] for r in rows])

    layouts = {}
    for spec in args.run:
        name, path = spec.split(":", 1)
        run = load_scores(Path(path))
        missing = [k for k in keys if k not in run]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} of {len(keys)} windows missing")
        s = np.array([run[k]["score"] for k in keys])
        layouts[name] = s
        if args.gate is not None and name != "draft":
            hot = np.array([run[k]["hot_area"] for k in keys])
            layouts[f"{name}+gate"] = np.where(hot >= args.gate, draft_s, s)

    everything = np.arange(len(rows))
    rng = np.random.default_rng(SEED)
    draws = [bootstrap_index(rows, manifest, rng) for _ in range(DRAWS)]
    base = [metrics(draft_s, masks, d) for d in draws]
    result = {}
    for name, s in layouts.items():
        point = metrics(s, masks, everything)
        boot = [metrics(s, masks, d) for d in draws]
        entry = {}
        for t in TIERS:
            d_auc = np.array([b[t][0] - c[t][0] for b, c in zip(boot, base)])
            d_ap = np.array([b[t][1] - c[t][1] for b, c in zip(boot, base)])
            entry[t] = {"auc": point[t][0], "ap": point[t][1],
                        "d_auc": [float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))],
                        "d_ap": [float(np.percentile(d_ap, 2.5)), float(np.percentile(d_ap, 97.5))]}
        result[name] = entry
        print(f"{name:>8}  " + "  ".join(
            f"{t}: {100 * entry[t]['auc']:.1f}/{100 * entry[t]['ap']:.1f} "
            f"dAP[{100 * entry[t]['d_ap'][0]:+.1f},{100 * entry[t]['d_ap'][1]:+.1f}]" for t in TIERS), flush=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
