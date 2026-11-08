#!/usr/bin/env python3
"""Causal per-video calibration of the V-JEPA 2 gate score (2026-11-08). Offline, CPU only: no model runs.

V-JEPA 2's holmes head ranks windows well inside a video (within AUC 0.805) but its score level differs between videos.
Each calibration uses only the past windows of the same video (streaming-safe):
  raw            the head score
  minus_mean_k   score - mean of the previous k windows (k = all: every previous window)
  zscore_k       (score - mean) / std of the previous k windows (needs >= 2 previous windows)
A window without enough past gets -inf: it is never up-scaled and ranks last when the score is used alone.

Gate: the VLM uses its 2x score when the calibrated score is in the top 20% / 30% of the 1,684 windows (read on test).
Reuses holmes_scores, window_groups and evaluate from results/spark/vjepa_gate_ucf117/20261108-062221/cascade.py.

  uv run --frozen python results/spark/vjepa_gate_calib_ucf117/<run>/calibrate.py --config <run>/config.yaml --out-dir <run>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[4]
CASCADE = REPO / "results/spark/vjepa_gate_ucf117/20261108-062221/cascade.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def minus_mean(past: np.ndarray, score: float) -> float:
    return score - past.mean() if len(past) >= 1 else -np.inf


def zscore(past: np.ndarray, score: float) -> float:
    if len(past) < 2:
        return -np.inf
    return (score - past.mean()) / max(past.std(), 1e-6)


CALIBRATIONS = {"minus_mean": minus_mean, "zscore": zscore}


def calibrate(keys: list[tuple[str, int]], raw: np.ndarray, kind: str, k: int | None) -> np.ndarray:
    """Calibrated score per window; keys sorted by (video, window). k None = every previous window."""
    out = np.empty(len(raw))
    start = 0
    for i, (video, _) in enumerate(keys):
        if i > 0 and keys[i - 1][0] != video:
            start = i
        lo = start if k is None else max(start, i - k)
        out[i] = CALIBRATIONS[kind](raw[lo:i], raw[i])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    le = load_module("layout_eval", REPO / "scripts/layout-eval.py")
    cascade = load_module("cascade", CASCADE)

    manifest = {m["name"]: m for m in json.loads((REPO / config["manifest"]).read_text())}
    events = [json.loads(line) for line in (REPO / config["events"]).read_text().splitlines()]
    gate_by_key, state = cascade.holmes_scores(Path(config["gate"]["fit_dir"]), config["gate"]["method"],
                                               config["gate"]["kind"])
    keys = sorted(gate_by_key)
    raw = np.array([gate_by_key[k] for k in keys])
    gates = {"raw": raw}
    for kind, k in config["calibrations"]:
        gates[f"{kind}_{'all' if k is None else k}"] = calibrate(keys, raw, kind, k)
    print(f"gate layer {state}; {len(keys)} windows", flush=True)

    rates = {}
    for model, paths in config["models"].items():
        orig_rows, zoom_rows = le.load_scores(REPO / paths["original"]), le.load_scores(REPO / paths["upscaled"])
        rows = [orig_rows[k] for k in keys]
        orig = np.array([r["score"] for r in rows])
        zoom = np.array([zoom_rows[k]["score"] for k in keys])
        masks = le.tier_masks(rows, manifest, events)
        groups = cascade.window_groups(rows, masks, manifest)
        layouts = {"original": orig, "upscaled all": zoom}
        rates[model] = {}
        for name, g in gates.items():
            if model == next(iter(config["models"])):
                layouts[f"{name} alone"] = np.where(np.isfinite(g), g, g[np.isfinite(g)].min() - 1)
            for rate in config["upscale_rates"]:
                zoomed = g >= np.quantile(g[np.isfinite(g)], 1 - rate * len(g) / np.isfinite(g).sum())
                label = f"{name} top {int(100 * rate)}%"
                layouts[label] = np.where(zoomed, zoom, orig)
                rates[model][label] = {grp: float(zoomed[m].mean()) for grp, m in groups.items()} | \
                                      {"all windows": float(zoomed.mean())}
        print(f"== {model}", flush=True)
        result = cascade.evaluate(le, rows, masks, manifest, orig, layouts)
        (args.out_dir / f"eval-{model}.json").write_text(json.dumps(result, indent=1))
    for label, r in rates[next(iter(rates))].items():
        print(f"{label:>22}  up-scaled: " + ", ".join(f"{g} {100 * v:.1f}%" for g, v in r.items()), flush=True)
    (args.out_dir / "rates.json").write_text(json.dumps(rates, indent=1))


if __name__ == "__main__":
    main()
