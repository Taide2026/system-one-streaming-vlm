#!/usr/bin/env python3
"""Split the V-JEPA 2 gate result on 290 UCF test videos into the 117 short videos and the other 173 (2026-11-08).

Question: the gate gain is all dAP +3.7 on the 117 short videos (vjepa_gate_calib_ucf117) but +1.3 on 290
(vjepa_gate_ucf290), at the same setting (zscore_all, top 20%). Is the gain smaller on the 173 other videos, and where
does the up-scale budget go there? Offline, CPU only. Same gate, calibration and metrics as vjepa_gate_ucf290.

Two threshold rules:
- global: the top 20% of all 18,390 windows (the rule of vjepa_gate_ucf290), then evaluated per subset;
- per subset: the top 20% of the windows of that subset only.

  uv run --frozen python results/spark/vjepa_gate_split_ucf290/<run>/split.py --config <run>/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
GATE290 = REPO / "results/spark/vjepa_gate_ucf290/20261108-064224/gate290.py"
RATE = 0.2


def top(g: np.ndarray, rate: float) -> np.ndarray:
    """gate290 rule: zoom the top rate * len(g) windows; -inf windows (too little past) never zoom."""
    finite = np.isfinite(g)
    return g >= np.quantile(g[finite], 1 - rate * len(g) / finite.sum())


def subset_report(le, cascade, rows, manifest, events, orig, zoom, label, zoomed: dict[str, np.ndarray]) -> dict:
    """Point AP, bootstrap dAP and zoom rates on one subset of rows."""
    masks = le.tier_masks(rows, manifest, events)
    groups = cascade.window_groups(rows, masks, manifest)
    layouts = {"upscaled all": zoom, "oracle (events only)": np.where(label, zoom, orig)}
    layouts |= {f"gate, {rule}": np.where(z, zoom, orig) for rule, z in zoomed.items()}
    result = cascade.evaluate(le, rows, masks, manifest, orig, {"original": orig} | layouts)
    rates = {rule: {grp: float(z[m].mean()) for grp, m in groups.items()} | {"all windows": float(z.mean())}
             for rule, z in zoomed.items()}
    for rule, r in rates.items():
        print(f"{'zoomed, ' + rule:>26}: " + ", ".join(f"{g} {100 * v:.1f}%" for g, v in r.items()), flush=True)
    sizes = {grp: int(m.sum()) for grp, m in groups.items()}
    print(f"{'windows':>26}: " + ", ".join(f"{g} {n}" for g, n in sizes.items()), flush=True)
    return {"eval": result, "rates": rates, "windows": sizes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    g290 = __import__("importlib.util").util
    spec = g290.spec_from_file_location("gate290", GATE290)
    gate290 = g290.module_from_spec(spec)
    spec.loader.exec_module(gate290)
    le = gate290.load_module("layout_eval", REPO / "scripts/layout-eval.py")
    cascade = gate290.load_module("cascade", gate290.CASCADE)
    calib = gate290.load_module("calibrate", gate290.CALIBRATE)

    manifest = {m["name"]: m for m in json.loads((REPO / config["manifest"]).read_text())}
    events = [json.loads(line) for line in (REPO / config["events"]).read_text().splitlines()]
    by_start = gate290.stored_gate(Path(config["gate"]["scores"]))
    short = {json.loads(line)["video"] for line in Path(config["short_videos"]).read_text().splitlines()}

    out = {}
    for model, paths in config["models"].items():
        orig_rows, zoom_rows = le.load_scores(REPO / paths["original"]), le.load_scores(REPO / paths["upscaled"])
        keys = sorted(orig_rows)
        rows = [orig_rows[k] for k in keys]
        raw = np.array([by_start[(r["video"], r["start_s"])] for r in rows])
        g = calib.calibrate(keys, raw, "zscore", None)
        orig = np.array([r["score"] for r in rows])
        zoom = np.array([zoom_rows[k]["score"] for k in keys])
        label = np.array([r["label"] for r in rows], dtype=bool)
        in_short = np.array([r["video"] in short for r in rows])
        zoomed_global = top(g, RATE)
        print(f"== {model}: zoomed windows {int(zoomed_global.sum())}, of them in the 117 short videos "
              f"{int(zoomed_global[in_short].sum())} ({100 * in_short.mean():.1f}% of all windows are short)",
              flush=True)
        out[model] = {}
        for name, sel in (("117 short", in_short), ("173 other", ~in_short)):
            idx = np.flatnonzero(sel)
            print(f"-- {model}, {name}: {len(idx)} windows", flush=True)
            zoomed = {"global top 20%": zoomed_global[idx], "per-subset top 20%": top(g[idx], RATE)}
            out[model][name] = subset_report(le, cascade, [rows[i] for i in idx], manifest, events,
                                             orig[idx], zoom[idx], label[idx], zoomed)
    (HERE / "split.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
