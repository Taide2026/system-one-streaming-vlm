#!/usr/bin/env python3
"""Score-based gates and score fusion for the 2x cascade (2026-11-09). Offline, CPU only.

Ideas from the Sol survey (2026-11-08-gate-selection-survey-sol.md), all decided on the 390 training videos:
  band    zoom when the 1x score is near a center; rank by -|s1 - center|, cutoff = 20% of train windows,
          center picked on train (and Sol's fixed 0.7 as a reference)
  rise    zoom when r_t = s1_t - mean(previous <= 16 s1 of the video) is in the top 20% of train (causal)
  motion  g_t = r_t + beta * m_t (m_t = train quantile of the TGM hot area), beta picked on train
  anomaly the V-JEPA 2 anomaly-head online gate (causal z-score >= 1.095, from vjepa_gate_online_ucf290)
Fusion on the zoomed windows: s1 + lambda (s2 - s1 [- delta]) or max(s1, s2), picked on train per gate.
Train labels = Holmes; test labels and tiers = scripts/layout-eval.py.

  uv run --frozen python results/spark/score_gate_ucf390/<run>/gates.py --config <run>/config.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
GATE290 = REPO / "results/spark/vjepa_gate_ucf290/20261108-064224/gate290.py"
ANOMALY_CUTOFF = 1.095  # vjepa_gate_online_ucf290/20261108-115929, train crossfit


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """scripts/layout-eval.py auc_ap formula (stable sort, no tie averaging)."""
    order = np.argsort(-s, kind="stable")
    yy = y[order]
    return float((np.cumsum(yy) / np.arange(1, len(yy) + 1) * yy).sum() / max(yy.sum(), 1))


# ---------- one split (train or test): windows in (video, window) order

class Split:
    def __init__(self, keys, y, s1, s2, hot, anomaly_z):
        self.keys, self.y, self.s1, self.s2, self.hot, self.anomaly_z = keys, y, s1, s2, hot, anomaly_z
        self.rise = rise(keys, s1, HISTORY)


HISTORY = 16


def rise(keys, s1: np.ndarray, history: int) -> np.ndarray:
    """r_t = s1_t - mean of the previous <= history 1x scores of the same video; 0 for the first window."""
    out = np.zeros(len(s1))
    for i, (video, _) in enumerate(keys):
        past = [j for j in range(max(0, i - history), i) if keys[j][0] == video]
        out[i] = s1[i] - s1[past].mean() if past else 0.0
    return out


def top_cutoff(scores: np.ndarray, rate: float) -> float:
    finite = scores[np.isfinite(scores)]
    return float(np.quantile(finite, 1 - rate * len(scores) / len(finite)))


# ---------- gates: each returns (train score, test score); zoom = score >= the train top-rate cutoff

def band_score(split: Split, center: float) -> np.ndarray:
    return -np.abs(split.s1 - center)


def motion_score(split: Split, beta: float, train_hot: np.ndarray) -> np.ndarray:
    m = np.searchsorted(np.sort(train_hot), split.hot, side="right") / len(train_hot)
    return split.rise + beta * m


def zoom_masks(train_s: np.ndarray, test_s: np.ndarray, rate: float, cutoff: float | None = None):
    c = top_cutoff(train_s, rate) if cutoff is None else cutoff
    return train_s >= c, test_s >= c


def gated_ap(split: Split, zoom: np.ndarray, fused: np.ndarray | None = None) -> float:
    return average_precision(split.y, np.where(zoom, split.s2 if fused is None else fused, split.s1))


# ---------- fusion

def fusions(cfg: dict, delta: float) -> dict[str, callable]:
    table = {"max": lambda s1, s2: np.maximum(s1, s2)}
    for lam in cfg["fusion_lambdas"]:
        table[f"lambda {lam}"] = lambda s1, s2, lam=lam: s1 + lam * (s2 - s1)
        table[f"lambda {lam}, delta"] = lambda s1, s2, lam=lam: s1 + lam * (s2 - s1 - delta)
    return table


def best_fusion(train: Split, zoom: np.ndarray, table: dict) -> str:
    return max(table, key=lambda n: gated_ap(train, zoom, table[n](train.s1, train.s2)))


# ---------- data

def load_train(cfg: dict, model: str, anomaly: dict) -> Split:
    root = REPO / cfg["train"]["scores"] / model
    s1 = {(r["video"], r["window"]): r for r in read_jsonl(root / "1x" / "scores.jsonl")}
    s2 = {(r["video"], r["window"]): r["score"] for r in read_jsonl(root / "2x" / "scores.jsonl")}
    hot = {(r["video"], r["window"]): r["hot_area"] for r in read_jsonl(REPO / cfg["train"]["windows"])}
    keys = sorted(s1)
    return Split(keys, np.array([anomaly[k]["holmes"] for k in keys]), np.array([s1[k]["score"] for k in keys]),
                 np.array([s2[k] for k in keys]), np.array([hot[k] for k in keys]),
                 np.array([anomaly[k]["z"] for k in keys]))


def anomaly_train(cfg: dict, calib) -> dict:
    rows = sorted(read_jsonl(cfg["train"]["anomaly_gate"]), key=lambda r: (r["video"], r["window"]))
    keys = [(r["video"], r["window"]) for r in rows]
    z = calib.calibrate(keys, np.array([r["crossfit"] for r in rows]), "zscore", None)
    return {k: {"holmes": r["holmes"], "z": zz} for k, r, zz in zip(keys, rows, z)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(p.parse_args().config.read_text())
    global HISTORY
    HISTORY = cfg["rise_history"]
    g290 = load_module("gate290", GATE290)
    le = g290.load_module("layout_eval", REPO / "scripts/layout-eval.py")
    cascade = g290.load_module("cascade", g290.CASCADE)
    calib = g290.load_module("calibrate", g290.CALIBRATE)
    manifest = {m["name"]: m for m in json.loads((REPO / cfg["test"]["manifest"]).read_text())}
    events = read_jsonl(REPO / cfg["test"]["events"])
    by_start = g290.stored_gate(Path(cfg["test"]["anomaly_gate"]))
    anomaly = anomaly_train(cfg, calib)
    rate = cfg["rate"]
    out = {}
    for model, paths in cfg["test"]["models"].items():
        train = load_train(cfg, model, anomaly)
        orig_rows, zoom_rows = le.load_scores(REPO / paths["original"]), le.load_scores(REPO / paths["upscaled"])
        keys = sorted(orig_rows)
        rows = [orig_rows[k] for k in keys]
        test_z = calib.calibrate(keys, np.array([by_start[(r["video"], r["start_s"])] for r in rows]), "zscore", None)
        test = Split(keys, np.array([r["label"] for r in rows]), np.array([r["score"] for r in rows]),
                     np.array([zoom_rows[k]["score"] for k in keys]), np.array([zoom_rows[k]["hot_area"] for k in keys]),
                     test_z)
        delta = float((train.s2 - train.s1)[train.y == 0].mean())
        table = fusions(cfg, delta)
        log = {"delta": delta, "train": {}}
        print(f"== {model}: train {len(train.y)} windows, test {len(test.y)} windows; delta {delta:+.4f}", flush=True)
        print(f"train AP: original {100*average_precision(train.y, train.s1):.2f}, all 2x "
              f"{100*average_precision(train.y, train.s2):.2f}, oracle "
              f"{100*average_precision(train.y, np.where(train.y == 1, train.s2, train.s1)):.2f}", flush=True)

        # Gates (train zoom, test zoom). Parameters picked on train gate AP (replacement).
        centers = {c: gated_ap(train, zoom_masks(band_score(train, c), band_score(test, c), rate)[0])
                   for c in cfg["band_centers"]}
        center = max(centers, key=centers.get)
        betas = {b: gated_ap(train, zoom_masks(motion_score(train, b, train.hot), motion_score(test, b, train.hot), rate)[0])
                 for b in cfg["motion_betas"]}
        beta = max(betas, key=betas.get)
        log["train"] |= {"band centers": centers, "band center": center, "motion betas": betas, "motion beta": beta}
        print("train band AP by center: " + ", ".join(f"{c}: {100*v:.2f}" for c, v in centers.items()), flush=True)
        print(f"band center picked: {center}; motion betas: " + ", ".join(f"{b}: {100*v:.2f}" for b, v in betas.items())
              + f"; beta picked: {beta}", flush=True)
        gates = {
            f"band, center {center} (train)": zoom_masks(band_score(train, center), band_score(test, center), rate),
            f"band, center {cfg['band_fixed_center']} (Sol, reference)": zoom_masks(
                band_score(train, cfg["band_fixed_center"]), band_score(test, cfg["band_fixed_center"]), rate),
            "rise": zoom_masks(train.rise, test.rise, rate),
            f"rise + motion, beta {beta}": zoom_masks(motion_score(train, beta, train.hot),
                                                      motion_score(test, beta, train.hot), rate),
            "anomaly head (online)": zoom_masks(train.anomaly_z, test.anomaly_z, rate, ANOMALY_CUTOFF),
            "all windows (reference)": (np.ones(len(train.y), bool), np.ones(len(test.y), bool)),
        }

        orig, zoom = test.s1, test.s2
        label = test.y.astype(bool)
        masks = le.tier_masks(rows, manifest, events)
        groups = cascade.window_groups(rows, masks, manifest)
        layouts = {"original": orig, "upscaled all": zoom, "oracle (events only)": np.where(label, zoom, orig)}
        rates = {}
        for name, (tr_zoom, te_zoom) in gates.items():
            fusion = best_fusion(train, tr_zoom, table)
            log["train"][name] = {"replacement": gated_ap(train, tr_zoom), "fusion": fusion,
                                  "fusion AP": gated_ap(train, tr_zoom, table[fusion](train.s1, train.s2)),
                                  "zoom rate": float(tr_zoom.mean())}
            print(f"train {name:42} zoom {100*tr_zoom.mean():5.1f}%  AP replace {100*log['train'][name]['replacement']:.2f}"
                  f"  fusion [{fusion}] {100*log['train'][name]['fusion AP']:.2f}", flush=True)
            if not name.startswith("all windows"):
                layouts[f"{name} | replace"] = np.where(te_zoom, zoom, orig)
            layouts[f"{name} | {fusion}"] = np.where(te_zoom, table[fusion](orig, zoom), orig)
            rates[name] = {g: float(te_zoom[m].mean()) for g, m in groups.items()} | {"all windows": float(te_zoom.mean())}
        print(f"-- test, {model}", flush=True)
        out[model] = {"log": log, "eval": cascade.evaluate(le, rows, masks, manifest, orig, layouts), "rates": rates}
        for name, r in rates.items():
            print(f"{name:>42}  up-scaled: " + ", ".join(f"{g} {100 * v:.1f}%" for g, v in r.items()), flush=True)
    (HERE / "gates.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
