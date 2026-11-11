#!/usr/bin/env python3
"""Zoom-benefit gate heads on V-JEPA 2 features (2026-11-08). CPU only, no model forward.

Label of a training window i (Holmes label h_i, VLM scores s1 at 1x and s2 at 2x):
  b_i = s2 - s1 if h_i = 1 else s1 - s2   (b > 0: the 2x up-scale helps this window)
Heads (linear, on standardized layer-18 patch_mean features), one per weight and per label model:
  ap        weighted BCE, y = [dAP_i > 0], weight |dAP_i| clipped at the 99th percentile; dAP_i = AP change of the
            training set when only window i uses its 2x score
  crossing  weighted BCE, y = [b_i > 0], weight = windows of the other class between s1_i and s2_i (AUC change)
  abs_b     weighted BCE, y = [b_i > 0], weight |b_i|
  logistic  BCE with pos_weight (as encoder-probing fit_window), y = [b_i > 0]
  ridge     MSE on b_i / std(b)
Training loop = encoder-probing s1stream.probe.fit._fit_linear (zero init, Adam, full batch, + lam |w|^2).
Selection (fixed before the run): 5-fold cross-validation by video on the training videos; held-out windows use
their fold head and a cutoff c = top 20% of the 4 training folds' scores; pooled held-out gate AP (all tier).
Test: head fit on all 390 videos; c = top 20% of the cross-fitted training scores; each head gates both VLMs.

  uv run --frozen python results/spark/benefit_gate_ucf390/<run>/benefit.py --config <run>/config.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
GATE290 = REPO / "results/spark/vjepa_gate_ucf290/20261108-064224/gate290.py"
MODELS = ("zeroshot", "mixkl")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# ---------- data

def features(folder: Path, names: list[str], cfg: dict) -> tuple[np.ndarray, list[dict]]:
    x = [np.asarray(np.load(folder / n / f"{cfg['kind']}.npy", mmap_mode="r")[:, cfg["state"]], dtype=np.float32)
         for n in names]
    meta = [m for n in names for m in read_jsonl(folder / n / "meta.jsonl")]
    return np.concatenate(x), meta


def scores_by_key(path: Path) -> dict[tuple[str, float], float]:
    return {(r["video"], r["start_s"]): float(r["score"]) for r in read_jsonl(path)}


def aligned(meta: list[dict], path: Path) -> np.ndarray:
    table = scores_by_key(path)
    missing = [m for m in meta if (m["video"], m["start_s"]) not in table]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} windows missing, first {missing[0]['video']} {missing[0]['start_s']}")
    return np.array([table[(m["video"], m["start_s"])] for m in meta])


# ---------- labels and weights

def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """scripts/layout-eval.py auc_ap formula (stable sort, no tie averaging)."""
    order = np.argsort(-s, kind="stable")
    yy = y[order]
    tp = np.cumsum(yy)
    return float((tp / np.arange(1, len(yy) + 1) * yy).sum() / max(yy.sum(), 1))


def single_dap(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """AP change when only window i uses its 2x score, all others at 1x (exact, one re-sort per window)."""
    base = average_precision(y, s1)
    out = np.empty(len(y))
    for i in range(len(y)):
        s = s1.copy()
        s[i] = s2[i]
        out[i] = average_precision(y, s) - base
    return out


def crossings(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Windows of the other class whose 1x score lies between s1_i and s2_i."""
    pos, neg = np.sort(s1[y == 1]), np.sort(s1[y == 0])
    lo, hi = np.minimum(s1, s2), np.maximum(s1, s2)
    between = lambda ref: np.searchsorted(ref, hi) - np.searchsorted(ref, lo)  # noqa: E731
    return np.where(y == 1, between(neg), between(pos)).astype(float)


def targets(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, clip: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{weight name: (label, weight)}; ridge gets (b / std(b), None), logistic (y, None)."""
    b = np.where(y == 1, s2 - s1, s1 - s2)
    dap = single_dap(y, s1, s2)
    ap_w = np.minimum(np.abs(dap), np.percentile(np.abs(dap), clip))
    up = (b > 0).astype(np.float32)
    return {"ap": ((dap > 0).astype(np.float32), ap_w), "crossing": (up, crossings(y, s1, s2)),
            "abs_b": (up, np.abs(b)), "logistic": (up, None), "ridge": ((b / b.std()).astype(np.float32), None)}


# ---------- heads

def standardize(train: np.ndarray, *others: np.ndarray) -> list[torch.Tensor]:
    """= encoder-probing s1stream.probe.fit.standardize."""
    mean, std = train.mean(axis=0), train.std(axis=0)
    return [torch.tensor((a - mean) / (std + 1e-4), dtype=torch.float32) for a in (train, *others)]


def fit_linear(x: torch.Tensor, loss_fn, cfg: dict) -> torch.nn.Linear:
    """= encoder-probing s1stream.probe.fit._fit_linear on the CPU."""
    torch.manual_seed(cfg["seed"])
    head = torch.nn.Linear(x.shape[1], 1)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam(head.parameters(), lr=cfg["lr"])
    for _ in range(cfg["steps"]):
        optimizer.zero_grad()
        loss = loss_fn(head(x).squeeze(1)) + cfg["lam"] * head.weight.square().sum()
        loss.backward()
        optimizer.step()
    return head


def loss_for(name: str, label: np.ndarray, weight: np.ndarray | None):
    t = torch.tensor(label, dtype=torch.float32)
    if name == "ridge":
        return lambda s: torch.mean((s - t) ** 2)
    if name == "logistic":
        pos = t.mean().clamp(min=1e-6)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).detach())
        return lambda s: bce(s, t)
    w = torch.tensor(weight / weight.mean(), dtype=torch.float32)
    return lambda s: torch.mean(w * torch.nn.functional.binary_cross_entropy_with_logits(s, t, reduction="none"))


def fit_score(x_fit: np.ndarray, label: np.ndarray, weight, name: str, cfg: dict, *x_score: np.ndarray) -> list[np.ndarray]:
    """Fit on x_fit (standardized on itself); return the scores of x_fit and of each x_score."""
    tensors = standardize(x_fit, *x_score)
    head = fit_linear(tensors[0], loss_for(name, label, weight), cfg)
    with torch.inference_mode():
        return [head(t).squeeze(1).numpy() for t in tensors]


def top_cutoff(scores: np.ndarray, rate: float) -> float:
    return float(np.quantile(scores, 1 - rate))


# ---------- main

def folds_of(videos: list[str], cfg: dict) -> np.ndarray:
    names = sorted(set(videos))
    fold = dict(zip(names, np.random.default_rng(cfg["seed"]).permutation(len(names)) % cfg["n"]))
    return np.array([fold[v] for v in videos])


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--config", type=Path, required=True)
    ap_.add_argument("--models", nargs="+", default=list(MODELS), help="label models (a check run uses one)")
    args = ap_.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    g290 = load_module("gate290", GATE290)
    le = g290.load_module("layout_eval", REPO / "scripts/layout-eval.py")
    cascade = g290.load_module("cascade", g290.CASCADE)
    rate = cfg["upscale_rate"]

    x_tr, meta_tr = features(Path(cfg["train"]["features"]), ["train_anomaly", "train_normal"], cfg["feature"])
    y_tr = np.array([m["holmes"] for m in meta_tr])
    fold = folds_of([m["video"] for m in meta_tr], cfg["folds"])
    x_te = np.concatenate([features(Path(p), [""], cfg["feature"])[0] for p in cfg["test"]["features"]])
    meta_te = [m for p in cfg["test"]["features"] for m in read_jsonl(Path(p) / "meta.jsonl")]
    print(f"train {len(y_tr)} windows ({int(y_tr.sum())} Holmes events), test {len(meta_te)} windows", flush=True)

    # Train-side: labels, cross-validation, cross-fitted cutoff, final test scores.
    heads, cv = {}, {}
    for label_model in args.models:
        s1 = aligned(meta_tr, REPO / cfg["train"]["scores"] / label_model / "1x" / "scores.jsonl")
        s2 = aligned(meta_tr, REPO / cfg["train"]["scores"] / label_model / "2x" / "scores.jsonl")
        tg = targets(y_tr, s1, s2, cfg["ap_clip_percentile"])
        cv[label_model] = {"original": average_precision(y_tr, s1), "upscaled all": average_precision(y_tr, s2),
                           "oracle": average_precision(y_tr, np.where(y_tr == 1, s2, s1))}
        for name in cfg["weights"]:
            label, weight = tg[name]
            held = np.empty(len(y_tr))
            zoom = np.zeros(len(y_tr), bool)
            for k in range(cfg["folds"]["n"]):
                tr, te = fold != k, fold == k
                fit_s, held_s = fit_score(x_tr[tr], label[tr], None if weight is None else weight[tr], name,
                                          cfg["head"], x_tr[te])
                held[te] = held_s
                zoom[te] = held_s >= top_cutoff(fit_s, rate)
            cv[label_model][name] = average_precision(y_tr, np.where(zoom, s2, s1))
            c = top_cutoff(held, rate)
            _, test_s = fit_score(x_tr, label, weight, name, cfg["head"], x_te)
            heads[(label_model, name)] = (test_s, c)
            print(f"cv {label_model:8} {name:8} gate AP {100 * cv[label_model][name]:.1f}; test zoom rate "
                  f"{100 * (test_s >= c).mean():.1f}%", flush=True)
        print(f"cv {label_model:8} original {100*cv[label_model]['original']:.1f}, all 2x "
              f"{100*cv[label_model]['upscaled all']:.1f}, oracle {100*cv[label_model]['oracle']:.1f}", flush=True)
    main_weight = {m: max(cfg["weights"], key=lambda n: cv[m][n]) for m in args.models}
    print(f"main (highest cv gate AP): {main_weight}", flush=True)

    # Test-side: every head gates both VLMs.
    manifest = {m["name"]: m for m in json.loads((REPO / cfg["test"]["manifest"]).read_text())}
    events = read_jsonl(REPO / cfg["test"]["events"])
    test_key = {(m["video"], m["start_s"]): i for i, m in enumerate(meta_te)}
    out = {"cv": cv, "main": main_weight}
    for eval_model, paths in cfg["test"]["models"].items():
        orig_rows, zoom_rows = le.load_scores(REPO / paths["original"]), le.load_scores(REPO / paths["upscaled"])
        keys = sorted(orig_rows)
        rows = [orig_rows[k] for k in keys]
        idx = np.array([test_key[(r["video"], r["start_s"])] for r in rows])
        orig = np.array([r["score"] for r in rows])
        zoom = np.array([zoom_rows[k]["score"] for k in keys])
        label = np.array([r["label"] for r in rows], dtype=bool)
        masks = le.tier_masks(rows, manifest, events)
        groups = cascade.window_groups(rows, masks, manifest)
        layouts = {"original": orig, "upscaled all": zoom, "oracle (events only)": np.where(label, zoom, orig)}
        rates = {}
        for (label_model, name), (test_s, c) in heads.items():
            zoomed = test_s[idx] >= c
            tag = f"{label_model}-label {name}"
            layouts[tag] = np.where(zoomed, zoom, orig)
            rates[tag] = {g: float(zoomed[m].mean()) for g, m in groups.items()} | {"all windows": float(zoomed.mean())}
        print(f"== test, gating {eval_model}: {len(rows)} windows", flush=True)
        out[eval_model] = {"eval": cascade.evaluate(le, rows, masks, manifest, orig, layouts), "rates": rates}
        for tag, r in rates.items():
            print(f"{tag:>26}  up-scaled: " + ", ".join(f"{g} {100 * v:.1f}%" for g, v in r.items()), flush=True)
    (HERE / "benefit.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
