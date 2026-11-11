#!/usr/bin/env python3
"""V-JEPA 2 gate on 290 UCF test videos with an online threshold (2026-11-08). Offline, CPU only.

vjepa_gate_ucf290 took the top 20% of all test windows; that threshold needs every test window, so it is not online.
Here each window is up-scaled when its causal z-score (all previous windows of the same video) is >= a fixed cutoff c.
c never comes from the test videos:
- train crossfit (main, fixed before the run): c up-scales 20% of the training windows, from the 5-fold
  cross-fitted head scores of the 390 training videos, calibrated in the same way;
- train insample: the same with the head that scores the test windows, on its own training windows;
- fixed c = 0.5, 1.0, 1.5 (a curve; picking its best point would use test).
The actual up-scale rate on test is reported for every rule.

  uv run --frozen python results/spark/vjepa_gate_online_ucf290/<run>/online.py --config <run>/config.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
GATE290 = REPO / "results/spark/vjepa_gate_ucf290/20261108-064224/gate290.py"


def load_gate290():
    spec = importlib.util.spec_from_file_location("gate290", GATE290)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_cutoff(calib, rows: list[dict], column: str, rate: float) -> float:
    """The z-score cutoff that up-scales `rate` of the training windows (windows without enough past never zoom)."""
    rows = sorted(rows, key=lambda r: (r["video"], r["window"]))
    keys = [(r["video"], r["window"]) for r in rows]
    z = calib.calibrate(keys, np.array([r[column] for r in rows]), "zscore", None)
    finite = np.isfinite(z)
    return float(np.quantile(z[finite], 1 - rate * len(z) / finite.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    g290 = load_gate290()
    le = g290.load_module("layout_eval", REPO / "scripts/layout-eval.py")
    cascade = g290.load_module("cascade", g290.CASCADE)
    calib = g290.load_module("calibrate", g290.CALIBRATE)

    train = [json.loads(line) for line in Path(config["train_scores"]).read_text().splitlines()]
    cutoffs = {f"train crossfit (c={c:.3f})": c for c in [train_cutoff(calib, train, "crossfit", config["train_rate"])]}
    cutoffs |= {f"train insample (c={c:.3f})": c for c in [train_cutoff(calib, train, "insample", config["train_rate"])]}
    cutoffs |= {f"fixed c={c}": float(c) for c in config["fixed_cutoffs"]}
    print("cutoffs: " + ", ".join(cutoffs), flush=True)

    manifest = {m["name"]: m for m in json.loads((REPO / config["manifest"]).read_text())}
    events = [json.loads(line) for line in (REPO / config["events"]).read_text().splitlines()]
    by_start = g290.stored_gate(Path(config["gate"]["scores"]))
    out = {"cutoffs": cutoffs}
    for model, paths in config["models"].items():
        orig_rows, zoom_rows = le.load_scores(REPO / paths["original"]), le.load_scores(REPO / paths["upscaled"])
        keys = sorted(orig_rows)
        rows = [orig_rows[k] for k in keys]
        z = calib.calibrate(keys, np.array([by_start[(r["video"], r["start_s"])] for r in rows]), "zscore", None)
        orig = np.array([r["score"] for r in rows])
        zoom = np.array([zoom_rows[k]["score"] for k in keys])
        label = np.array([r["label"] for r in rows], dtype=bool)
        masks = le.tier_masks(rows, manifest, events)
        groups = cascade.window_groups(rows, masks, manifest)
        layouts = {"original": orig, "upscaled all": zoom, "oracle (events only)": np.where(label, zoom, orig)}
        rates = {}
        for name, c in cutoffs.items():
            zoomed = z >= c  # -inf (too little past) never zooms
            layouts[name] = np.where(zoomed, zoom, orig)
            rates[name] = {grp: float(zoomed[m].mean()) for grp, m in groups.items()} | {"all windows": float(zoomed.mean())}
        print(f"== {model}: {len(rows)} windows", flush=True)
        out[model] = {"eval": cascade.evaluate(le, rows, masks, manifest, orig, layouts), "rates": rates}
    for name, r in out[next(iter(config["models"]))]["rates"].items():
        print(f"{name:>28}  up-scaled: " + ", ".join(f"{g} {100 * v:.1f}%" for g, v in r.items()), flush=True)
    (HERE / "online.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
