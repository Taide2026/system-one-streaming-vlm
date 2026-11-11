#!/usr/bin/env python3
"""Zoom-benefit gate heads, input = V-JEPA 2 features + the VLM 1x score (2026-11-08). CPU only.

Question: does the gate pick better windows when it also sees the VLM's own 1x score? The cascade runs 1x on
every window anyway, so this input is free and causal.
Everything else (labels, weights, training loop, 5-fold selection, online cutoff, test evaluation) = benefit.py
in this folder (copy of benefit_gate_ucf390/20261108-145639/benefit.py). The 1x score is one extra column,
standardized with the features. A head trained on label model M reads M's 1x score, so it gates only M.
Train 1x scores: Colab G4. Test 1x scores: GB10 (the machine difference is at most 0.027).

  uv run --frozen python results/spark/benefit_s1_gate_ucf390/<run>/benefit_s1.py --config <run>/config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import benefit as B

HERE = Path(__file__).resolve().parent


def with_score(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.concatenate([x, s[:, None].astype(np.float32)], axis=1)


def test_1x(meta_te: list[dict], path: Path) -> np.ndarray:
    table = {(r["video"], r["start_s"]): float(r["score"]) for r in B.read_jsonl(path)}
    return np.array([table[(m["video"], m["start_s"])] for m in meta_te])


def cross_validate(x, y, label, weight, name, fold, s1, s2, cfg):
    """Pooled held-out gate AP and the cross-fitted scores (for the online cutoff)."""
    held = np.empty(len(y))
    zoom = np.zeros(len(y), bool)
    for k in range(cfg["folds"]["n"]):
        tr, te = fold != k, fold == k
        fit_s, held_s = B.fit_score(x[tr], label[tr], None if weight is None else weight[tr], name, cfg["head"], x[te])
        held[te] = held_s
        zoom[te] = held_s >= B.top_cutoff(fit_s, cfg["upscale_rate"])
    return B.average_precision(y, np.where(zoom, s2, s1)), held


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(p.parse_args().config.read_text())
    REPO = B.REPO
    g290 = B.load_module("gate290", B.GATE290)
    le = g290.load_module("layout_eval", REPO / "scripts/layout-eval.py")
    cascade = g290.load_module("cascade", g290.CASCADE)

    x_tr, meta_tr = B.features(Path(cfg["train"]["features"]), ["train_anomaly", "train_normal"], cfg["feature"])
    y_tr = np.array([m["holmes"] for m in meta_tr])
    fold = B.folds_of([m["video"] for m in meta_tr], cfg["folds"])
    x_te = np.concatenate([B.features(Path(q), [""], cfg["feature"])[0] for q in cfg["test"]["features"]])
    meta_te = [m for q in cfg["test"]["features"] for m in B.read_jsonl(Path(q) / "meta.jsonl")]
    manifest = {m["name"]: m for m in json.loads((REPO / cfg["test"]["manifest"]).read_text())}
    events = B.read_jsonl(REPO / cfg["test"]["events"])
    test_key = {(m["video"], m["start_s"]): i for i, m in enumerate(meta_te)}
    print(f"train {len(y_tr)} windows, test {len(meta_te)} windows, input dim {x_tr.shape[1] + 1}", flush=True)

    out = {"cv": {}, "main": {}}
    for model, paths in cfg["test"]["models"].items():
        root = REPO / cfg["train"]["scores"] / model
        s1, s2 = B.aligned(meta_tr, root / "1x" / "scores.jsonl"), B.aligned(meta_tr, root / "2x" / "scores.jsonl")
        xs_tr, xs_te = with_score(x_tr, s1), with_score(x_te, test_1x(meta_te, REPO / paths["original"]))
        tg = B.targets(y_tr, s1, s2, cfg["ap_clip_percentile"])
        cv = {"original": B.average_precision(y_tr, s1), "upscaled all": B.average_precision(y_tr, s2),
              "oracle": B.average_precision(y_tr, np.where(y_tr == 1, s2, s1))}
        heads = {}
        for name in cfg["weights"]:
            label, weight = tg[name]
            cv[name], held = cross_validate(xs_tr, y_tr, label, weight, name, fold, s1, s2, cfg)
            _, test_s = B.fit_score(xs_tr, label, weight, name, cfg["head"], xs_te)
            heads[name] = (test_s, B.top_cutoff(held, cfg["upscale_rate"]))
            print(f"cv {model:8} {name:8} gate AP {100 * cv[name]:.2f}; test zoom rate "
                  f"{100 * (test_s >= heads[name][1]).mean():.1f}%", flush=True)
        print(f"cv {model:8} original {100*cv['original']:.1f}, all 2x {100*cv['upscaled all']:.1f}, "
              f"oracle {100*cv['oracle']:.1f}", flush=True)
        out["cv"][model] = cv
        out["main"][model] = max(cfg["weights"], key=lambda n: cv[n])
        print(f"main {model}: {out['main'][model]}", flush=True)

        orig_rows, zoom_rows = le.load_scores(REPO / paths["original"]), le.load_scores(REPO / paths["upscaled"])
        keys = sorted(orig_rows)
        rows = [orig_rows[k] for k in keys]
        idx = np.array([test_key[(r["video"], r["start_s"])] for r in rows])
        orig = np.array([r["score"] for r in rows])
        zoom = np.array([zoom_rows[k]["score"] for k in keys])
        lab = np.array([r["label"] for r in rows], dtype=bool)
        masks = le.tier_masks(rows, manifest, events)
        groups = cascade.window_groups(rows, masks, manifest)
        layouts = {"original": orig, "upscaled all": zoom, "oracle (events only)": np.where(lab, zoom, orig)}
        rates = {}
        for name, (test_s, c) in heads.items():
            zoomed = test_s[idx] >= c
            layouts[f"{model}-label {name} + 1x"] = np.where(zoomed, zoom, orig)
            rates[name] = {g: float(zoomed[m].mean()) for g, m in groups.items()} | {"all windows": float(zoomed.mean())}
        print(f"== test, gating {model}: {len(rows)} windows", flush=True)
        out[model] = {"eval": cascade.evaluate(le, rows, masks, manifest, orig, layouts), "rates": rates}
        for name, r in rates.items():
            print(f"{name:>10}  up-scaled: " + ", ".join(f"{g} {100 * v:.1f}%" for g, v in r.items()), flush=True)
    (HERE / "benefit_s1.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
