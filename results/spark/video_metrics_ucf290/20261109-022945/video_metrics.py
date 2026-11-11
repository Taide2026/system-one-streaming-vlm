#!/usr/bin/env python3
"""Video-level metrics (Macro AUC, within_auc, AUC_A, recall at 1 FA/h, frame AUC / AP) of the up-scale cascade.

The score layouts (original, 2x everywhere, oracle, every score gate with replacement and with its fusion) are rebuilt
with the functions of score_gate_ucf390/20261108-155845/gates.py and its config, the same way as its main().
Metrics and intervals: s1stream.metrics and s1stream.bootstrap.metric_summary of experiment/encoder-probing.

  uv run --frozen python results/spark/video_metrics_ucf290/<run>/video_metrics.py --config <run>/config.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def window_range(start_s: float, end_s: float, fps: float) -> tuple[int, int]:
    """s1stream.windows.window_frame_range: inclusive (round(start_s * fps), round(end_s * fps) - 1)."""
    return round(start_s * fps), round(end_s * fps) - 1


def frame_videos(rows: list[dict], scores: np.ndarray, manifest: dict) -> list[dict]:
    """metrics.frame_auc input, one dict per video in manifest order; rows and scores are aligned."""
    videos = {name: {"n_frames": m["n_frames"], "events": m["events"], "windows": []} for name, m in manifest.items()}
    for row, score in zip(rows, scores):
        fps = manifest[row["video"]]["fps"]
        videos[row["video"]]["windows"].append((*window_range(row["start_s"], row["end_s"], fps), float(score)))
    return [v for v in videos.values() if v["windows"]]


def build_layouts(gates, cfg: dict, model: str, ctx: dict) -> tuple[list[dict], dict[str, np.ndarray]]:
    """The test layouts of one VLM, as gates.main() builds them (same gates, cutoffs and train-picked fusions)."""
    rate = cfg["rate"]
    paths = cfg["test"]["models"][model]
    train = gates.load_train(cfg, model, ctx["anomaly"])
    orig_rows, zoom_rows = ctx["le"].load_scores(REPO / paths["original"]), ctx["le"].load_scores(REPO / paths["upscaled"])
    keys = sorted(orig_rows)
    rows = [orig_rows[k] for k in keys]
    z = ctx["calib"].calibrate(keys, np.array([ctx["by_start"][(r["video"], r["start_s"])] for r in rows]), "zscore", None)
    test = gates.Split(keys, np.array([r["label"] for r in rows]), np.array([r["score"] for r in rows]),
                       np.array([zoom_rows[k]["score"] for k in keys]),
                       np.array([zoom_rows[k]["hot_area"] for k in keys]), z)
    delta = float((train.s2 - train.s1)[train.y == 0].mean())
    table = gates.fusions(cfg, delta)

    def pick(values: dict) -> float:
        return max(values, key=values.get)

    center = pick({c: gates.gated_ap(train, gates.zoom_masks(gates.band_score(train, c), gates.band_score(test, c),
                                                             rate)[0]) for c in cfg["band_centers"]})
    beta = pick({b: gates.gated_ap(train, gates.zoom_masks(gates.motion_score(train, b, train.hot),
                                                           gates.motion_score(test, b, train.hot), rate)[0])
                 for b in cfg["motion_betas"]})
    fixed = cfg["band_fixed_center"]
    masks = {
        f"band, center {center} (train)": gates.zoom_masks(gates.band_score(train, center),
                                                           gates.band_score(test, center), rate),
        f"band, center {fixed} (Sol)": gates.zoom_masks(gates.band_score(train, fixed), gates.band_score(test, fixed), rate),
        "rise": gates.zoom_masks(train.rise, test.rise, rate),
        f"rise + motion, beta {beta}": gates.zoom_masks(gates.motion_score(train, beta, train.hot),
                                                        gates.motion_score(test, beta, train.hot), rate),
        "anomaly head (online)": gates.zoom_masks(train.anomaly_z, test.anomaly_z, rate, gates.ANOMALY_CUTOFF),
        "all windows": (np.ones(len(train.y), bool), np.ones(len(test.y), bool)),
    }
    orig, zoom = test.s1, test.s2
    layouts = {"original": orig, "2x everywhere": zoom, "oracle (events only)": np.where(test.y == 1, zoom, orig)}
    for name, (tr_zoom, te_zoom) in masks.items():
        fusion = gates.best_fusion(train, tr_zoom, table)
        if name != "all windows":
            layouts[f"{name}, replace"] = np.where(te_zoom, zoom, orig)
        layouts[f"{name}, {fusion}"] = np.where(te_zoom, table[fusion](orig, zoom), orig)
    return rows, layouts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(p.parse_args().config.read_text())
    sys.path.insert(0, cfg["metrics_code"])
    from s1stream import bootstrap  # encoder-probing copy (the small-event-fov branch has no bootstrap.py)

    gate_run = REPO / cfg["score_gate_run"]
    gates = load("gates", gate_run / "gates.py")
    gate_cfg = yaml.safe_load((gate_run / "config.yaml").read_text())
    gates.HISTORY = gate_cfg["rise_history"]
    g290 = gates.load_module("gate290", gates.GATE290)
    ctx = {"le": g290.load_module("layout_eval", REPO / "scripts/layout-eval.py"),
           "calib": g290.load_module("calibrate", g290.CALIBRATE),
           "by_start": g290.stored_gate(Path(gate_cfg["test"]["anomaly_gate"]))}
    ctx["anomaly"] = gates.anomaly_train(gate_cfg, ctx["calib"])
    manifest = {m["name"]: m for m in json.loads((REPO / cfg["manifest"]).read_text())}
    fps_values = {m["fps"] for m in manifest.values()}
    if len(fps_values) != 1:
        raise ValueError(f"Manifest videos must share one fps: {fps_values}")
    fps = fps_values.pop()

    out = {}
    for model in gate_cfg["test"]["models"]:
        rows, layouts = build_layouts(gates, gate_cfg, model, ctx)
        labels = np.array([r["label"] for r in rows])
        window_ap = {name: 100 * gates.average_precision(labels, s) for name, s in layouts.items()}
        runs = {name: frame_videos(rows, s, manifest) for name, s in layouts.items()}
        print(f"== {model}: {len(rows)} windows, {len(runs['original'])} videos", flush=True)
        summary = bootstrap.metric_summary(runs, fps, baseline=cfg["baseline"],
                                           n=cfg["bootstrap"]["n"], seed=cfg["bootstrap"]["seed"])
        out[model] = {"window_ap": window_ap} | summary
        for name in layouts:
            m = summary["runs"][name]
            print(f"{name:44} winAP {window_ap[name]:5.1f}  " + "  ".join(
                f"{k} {100 * m[k]['value']:.1f}" for k in bootstrap.VIDEO_METRICS), flush=True)
    (HERE / "video_metrics.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
