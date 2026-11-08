#!/usr/bin/env python3
"""V-JEPA 2 linear-head gate for the 2x up-scale (2026-11-08). Offline, CPU only: no model runs.

Cascade: window t takes the VLM's 2x score when the V-JEPA 2 holmes score of window t is >= a threshold; else it keeps
the VLM's original score. The holmes head (patch_mean, layer chosen on validation as in s1stream.encoder.compare)
comes from the encoder-probing run; its test set is 117 short UCF test videos, all inside the 290-video set.

  uv run --frozen python results/spark/vjepa_gate_ucf117/<run>/cascade.py --config <run>/config.yaml --out-dir <run>

Thresholds are quantiles of the gate score on these windows (up-scale rates 10/20/30/50%), read on the test set.
Tiers use the 290-video tier bounds (all 156 events are passed to tier_masks). Writes eval-<model>.json, rates.json.
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


def holmes_scores(fit_dir: Path, method: str, kind: str) -> tuple[dict, int]:
    """{(video, window): score} of the probe at its validation-selected layer, and that layer.
    Selection rule of s1stream.encoder.compare.select_state: holmes and knn the highest validation, mil the lowest."""
    fit = json.loads((fit_dir / "fit.json").read_text())
    values = [fit["choices"][f"{method}/{kind}/{s}"]["validation"] for s in range(fit["n_states"])]
    state = int(np.argmin(values)) if method == "mil" else int(np.argmax(values))
    scores = np.load(fit_dir / "test_scores.npy")[fit["methods"].index(method), fit["kinds"].index(kind), state]
    meta = [json.loads(line) for line in (fit_dir / "test_meta.jsonl").read_text().splitlines()]
    return {(m["video"], m["window"]): float(scores[m["i"]]) for m in meta}, state


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
    """layout-eval's point metrics and paired bootstrap of the AP change against orig, for each layout."""
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
        print(f"{name:>20}  " + "  ".join(
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
    gate_by_key, state = holmes_scores(Path(config["gate"]["fit_dir"]), config["gate"]["method"], config["gate"]["kind"])
    keys = sorted(gate_by_key)
    gate = np.array([gate_by_key[k] for k in keys])
    print(f"gate: {config['gate']['method']}/{config['gate']['kind']} layer {state}; {len(keys)} windows, "
          f"{len({k[0] for k in keys})} videos", flush=True)

    rates = {"gate_layer": state}
    for model, paths in config["models"].items():
        orig_rows, zoom_rows = le.load_scores(REPO / paths["original"]), le.load_scores(REPO / paths["upscaled"])
        rows = [orig_rows[k] for k in keys]
        orig = np.array([r["score"] for r in rows])
        zoom = np.array([zoom_rows[k]["score"] for k in keys])
        label = np.array([r["label"] for r in rows], dtype=bool)
        masks = le.tier_masks(rows, manifest, events)
        groups = window_groups(rows, masks, manifest)

        layouts = {"original": orig, "upscaled all": zoom, "oracle (events only)": np.where(label, zoom, orig),
                   "V-JEPA 2 head alone": gate}
        rates[model] = {}
        for rate in config["upscale_rates"]:
            threshold = float(np.quantile(gate, 1 - rate))
            zoomed = gate >= threshold
            name = f"gate top {int(100 * rate)}%"
            layouts[name] = np.where(zoomed, zoom, orig)
            rates[model][name] = {"threshold": threshold} | {g: float(zoomed[m].mean()) for g, m in groups.items()}
        print(f"== {model}", flush=True)
        result = evaluate(le, rows, masks, manifest, orig, layouts)
        (args.out_dir / f"eval-{model}.json").write_text(json.dumps(result, indent=1))
        for name, r in rates[model].items():
            print(f"{name:>20}  up-scaled: " + ", ".join(f"{g} {100 * v:.1f}%" for g, v in r.items() if g != "threshold"),
                  flush=True)
    (args.out_dir / "rates.json").write_text(json.dumps(rates, indent=1))


if __name__ == "__main__":
    main()
