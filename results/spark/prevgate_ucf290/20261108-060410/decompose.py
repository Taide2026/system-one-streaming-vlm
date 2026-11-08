#!/usr/bin/env python3
"""Where does the AP change of the 2x up-scale come from? (2026-11-08, offline, point estimates, no bootstrap)

For each tier, AUC / AP with event windows (positives) and negative windows each taking the original or the 2x score.
Uses the same inputs as gate.py (config.yaml) and the metrics of scripts/layout-eval.py.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
spec = importlib.util.spec_from_file_location("layout_eval", REPO / "scripts/layout-eval.py")
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)

config = yaml.safe_load((HERE / "config.yaml").read_text())
manifest = {m["name"]: m for m in json.loads((REPO / config["manifest"]).read_text())}
events = [json.loads(line) for line in (REPO / config["events"]).read_text().splitlines()]
for model, paths in config["models"].items():
    orig_rows, zoom_rows = le.load_scores(REPO / paths["original"]), le.load_scores(REPO / paths["upscaled"])
    keys = sorted(orig_rows)
    rows = [orig_rows[k] for k in keys]
    orig = np.array([r["score"] for r in rows])
    zoom = np.array([zoom_rows[k]["score"] for k in keys])
    label = np.array([r["label"] for r in rows], dtype=bool)
    masks = le.tier_masks(rows, manifest, events)
    everything = np.arange(len(rows))
    print(f"== {model}")
    for pos, neg in (("original", "original"), ("2x", "original"), ("original", "2x"), ("2x", "2x")):
        s = np.where(label, zoom if pos == "2x" else orig, zoom if neg == "2x" else orig)
        m = le.metrics(s, masks, everything)
        print(f"event windows {pos:>8}, negative windows {neg:>8}:  " +
              "  ".join(f"{t} {100 * m[t][0]:.1f}/{100 * m[t][1]:.1f}" for t in le.TIERS))
