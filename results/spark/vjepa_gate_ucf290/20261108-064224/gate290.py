#!/usr/bin/env python3
"""V-JEPA 2 linear-head gate for the 2x up-scale on all 290 UCF test videos (2026-11-08). Offline, CPU only.

Gate score: the V-JEPA 2 holmes head (patch_mean, layer 18, lam 0.01, refit on all training windows) on the 117 stored
short test videos plus the 173 videos extracted on 2026-11-08 (encoder-probing worktree,
results/spark/encoder_probe_ucf_test_rest_v1_f8/20261108-063724/refit-290/scores.jsonl). Rows mapped by (video, start_s).
Rule: the VLM uses its 2x score when the (calibrated) gate score is in the top X% of windows.
Fixed before the run (Eric, 2026-11-08): main = zscore over all previous windows, top 20%. Other rows are references.
Same code as results/spark/qwenvit_gate_ucf290/20261108-063536/gate290.py except the gate source.

  uv run --frozen python results/spark/vjepa_gate_ucf290/<run>/gate290.py --config <run>/config.yaml --out-dir <run>
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
CALIBRATE = REPO / "results/spark/vjepa_gate_calib_ucf117/20261108-063111/calibrate.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stored_gate(scores_jsonl: Path) -> dict[tuple[str, float], float]:
    """{(video, start_s): head score} from the refit scores.jsonl."""
    rows = [json.loads(line) for line in scores_jsonl.read_text().splitlines()]
    return {(r["video"], r["start_s"]): float(r["score"]) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    le = load_module("layout_eval", REPO / "scripts/layout-eval.py")
    cascade = load_module("cascade", CASCADE)
    calib = load_module("calibrate", CALIBRATE)

    manifest = {m["name"]: m for m in json.loads((REPO / config["manifest"]).read_text())}
    events = [json.loads(line) for line in (REPO / config["events"]).read_text().splitlines()]
    by_start = stored_gate(Path(config["gate"]["scores"]))

    rates = {}
    for model, paths in config["models"].items():
        orig_rows, zoom_rows = le.load_scores(REPO / paths["original"]), le.load_scores(REPO / paths["upscaled"])
        keys = sorted(orig_rows)
        rows = [orig_rows[k] for k in keys]
        raw = np.array([by_start[(r["video"], r["start_s"])] for r in rows])
        gates = {"raw": raw, "zscore_all": calib.calibrate(keys, raw, "zscore", None)}
        orig = np.array([r["score"] for r in rows])
        zoom = np.array([zoom_rows[k]["score"] for k in keys])
        label = np.array([r["label"] for r in rows], dtype=bool)
        masks = le.tier_masks(rows, manifest, events)
        groups = cascade.window_groups(rows, masks, manifest)
        layouts = {"original": orig, "upscaled all": zoom, "oracle (events only)": np.where(label, zoom, orig),
                   "head alone (raw)": raw}
        rates[model] = {}
        for name, g in gates.items():
            finite = np.isfinite(g)
            for rate in config["upscale_rates"]:
                zoomed = g >= np.quantile(g[finite], 1 - rate * len(g) / finite.sum())
                tag = f"{name} top {int(100 * rate)}%"
                layouts[tag] = np.where(zoomed, zoom, orig)
                rates[model][tag] = {grp: float(zoomed[m].mean()) for grp, m in groups.items()} | \
                                    {"all windows": float(zoomed.mean())}
        print(f"== {model}: {len(rows)} windows", flush=True)
        result = cascade.evaluate(le, rows, masks, manifest, orig, layouts)
        (args.out_dir / f"eval-{model}.json").write_text(json.dumps(result, indent=1))
    for tag, r in rates[next(iter(rates))].items():
        print(f"{tag:>22}  up-scaled: " + ", ".join(f"{g} {100 * v:.1f}%" for g, v in r.items()), flush=True)
    (args.out_dir / "rates.json").write_text(json.dumps(rates, indent=1))


if __name__ == "__main__":
    main()
