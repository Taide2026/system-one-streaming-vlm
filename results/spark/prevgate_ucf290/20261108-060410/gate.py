#!/usr/bin/env python3
"""Previous-window gate for the 2x up-scale (2026-11-08). Offline, CPU only: no model runs.

Window t of a video takes the up-scaled score when the score used at window t-1 lies in [lo, hi]; else it keeps
the original score. The score used at t-1 is the up-scaled one when t-1 was up-scaled (causal, as in streaming).
The first window of each video keeps the original score.

  uv run --frozen python results/spark/prevgate_ucf290/<run>/gate.py --config <run>/config.yaml --out-dir <run>

Writes eval-<model>.json (AUC / AP per tier, 95% paired bootstrap of the AP change against the original scores,
from scripts/layout-eval.py) and rates.json (share of windows up-scaled per window group).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[4]


def load_layout_eval():
    """scripts/layout-eval.py as a module (its file name has a hyphen)."""
    spec = importlib.util.spec_from_file_location("layout_eval", REPO / "scripts/layout-eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gated_scores(rows: list[dict], orig: np.ndarray, zoom: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """(score used per window, True where the window was up-scaled). rows are sorted by (video, window)."""
    used, zoomed = orig.copy(), np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        if i == 0 or rows[i - 1]["video"] != r["video"]:
            continue  # first window of a video: no signal yet
        if lo <= used[i - 1] <= hi:
            used[i], zoomed[i] = zoom[i], True
    return used, zoomed


def window_groups(rows: list[dict], masks: dict, manifest: dict) -> dict[str, np.ndarray]:
    """Boolean masks: normal windows of normal videos, normal windows of anomaly videos, event windows per tier."""
    label = np.array([r["label"] for r in rows], dtype=bool)
    normal_video = np.array([not manifest[r["video"]]["events"] for r in rows])
    groups = {"normal video": normal_video & ~label, "anomaly video, outside events": ~normal_video & ~label}
    for tier in ("small", "medium", "large"):
        keep, y = masks[tier]
        groups[f"event, {tier}"] = keep & y.astype(bool)
    return groups


def evaluate(le, rows, masks, manifest, orig, layouts: dict[str, np.ndarray]) -> dict:
    """layout-eval's point metrics and paired bootstrap against orig, for each layout."""
    everything = np.arange(len(rows))
    rng = np.random.default_rng(le.SEED)
    draws = [le.bootstrap_index(rows, manifest, rng) for _ in range(le.DRAWS)]
    base = [le.metrics(orig, masks, d) for d in draws]
    result = {}
    for name, s in layouts.items():
        point = le.metrics(s, masks, everything)
        boot = [le.metrics(s, masks, d) for d in draws]
        entry = {}
        for t in le.TIERS:
            d_ap = np.array([b[t][1] - c[t][1] for b, c in zip(boot, base)])
            entry[t] = {"auc": point[t][0], "ap": point[t][1],
                        "d_ap": [float(np.percentile(d_ap, 2.5)), float(np.percentile(d_ap, 97.5))]}
        result[name] = entry
        print(f"{name:>14}  " + "  ".join(
            f"{t}: {100 * entry[t]['auc']:.1f}/{100 * entry[t]['ap']:.1f} "
            f"dAP[{100 * entry[t]['d_ap'][0]:+.1f},{100 * entry[t]['d_ap'][1]:+.1f}]" for t in le.TIERS), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    le = load_layout_eval()

    manifest = {m["name"]: m for m in json.loads((REPO / config["manifest"]).read_text())}
    events = [json.loads(line) for line in (REPO / config["events"]).read_text().splitlines()]
    rates = {}
    for model, paths in config["models"].items():
        orig_rows = le.load_scores(REPO / paths["original"])
        zoom_rows = le.load_scores(REPO / paths["upscaled"])
        keys = sorted(orig_rows, key=lambda k: (k[0], k[1]))
        rows = [orig_rows[k] for k in keys]
        orig = np.array([r["score"] for r in rows])
        zoom = np.array([zoom_rows[k]["score"] for k in keys])
        masks = le.tier_masks(rows, manifest, events)
        groups = window_groups(rows, masks, manifest)

        layouts = {"original": orig, "upscaled all": zoom}
        rates[model] = {}
        for lo, hi in config["gates"]:
            name = f"prev in [{lo}, {hi}]"
            layouts[name], zoomed = gated_scores(rows, orig, zoom, lo, hi)
            rates[model][name] = {g: float(zoomed[m].mean()) for g, m in groups.items()} | {"all windows": float(zoomed.mean())}
        print(f"== {model}: {len(rows)} windows", flush=True)
        result = evaluate(le, rows, masks, manifest, orig, layouts)
        (args.out_dir / f"eval-{model}.json").write_text(json.dumps(result, indent=1))
        for name, r in rates[model].items():
            print(f"{name:>14}  up-scaled: " + ", ".join(f"{g} {100 * v:.1f}%" for g, v in r.items()), flush=True)
    (args.out_dir / "rates.json").write_text(json.dumps(rates, indent=1))


if __name__ == "__main__":
    main()
