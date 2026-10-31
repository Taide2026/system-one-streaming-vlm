"""Two score combinations from survey area12: fixed-weight fusion (Anomize) and a protected causal anchor (NOVA).

Question: the encoder head separates anomaly videos from normal videos but follows the scene; training-free optical
flow has no scene shortcut and some in-video ranking (motion_fit.py). Does a fixed combination keep both?
Scores (all causal):
  head          BCE head on the encoder feature x_t (window_eval.fit_bce; state from shortcut_fit.py bce)
  flow          training-free mean optical-flow magnitude (log1p), motion_fit.py
  fusion        (1 - w) * z(head) + w * z(flow), w in WEIGHTS                         (Anomize, fixed weights)
  anchor_dist   1 - cos(x_t, a_t); a_t = anchor of the earlier windows of the same video (NOVA VNA, causal):
                  a_t = (K0 * prior + sum_(i<t) e_i x_i) / (K0 + sum_(i<t) e_i),  e_i = exp(-z(head_i) / gamma)
                prior = mean train-normal feature; low-score (normal-looking) windows weigh more, so an event
                window adds little to the anchor; the first window of a video uses the prior only
  head_anchor   z(head) + lam * z(anchor_dist), lam in LAMBDAS, gamma in GAMMAS
z(): standardized with the mean and std of the validation windows. Features standardized with the train windows.
Selection: w, lam and gamma are chosen on the validation videos (probe.fit.split) by the pooled window AUC (Holmes
label > 0.5, all validation windows); the in-video AUC alone would always prefer flow, which cannot tell anomaly
videos from normal videos. The head for validation scores is fit on the fit split, the head for test on all train.
Test report: window_eval.evaluate; the test values of every weight are also written (not used for selection).
Usage: uv run python <this file> <encoder features run dir> <out name> [kind]   (default patch_mean)
Writes run-1/analysis-fusion/<out name>.json and prints one line per score.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from motion_fit import FEATURE_NAMES, MOTION_DIR
from pairwise_fit import fit_head, read_meta
from s1stream.probe.fit import split, standardize
from window_eval import NO_PAIRS, RUN, evaluate, line, previous_rows, test_items

WEIGHTS = tuple(round(0.1 * i, 1) for i in range(11))
LAMBDAS = (0.25, 0.5, 1.0, 2.0)
GAMMAS = (0.3, 1.0)
K0 = 1.0


def head_state(name: str, kind: str) -> tuple[int, float]:
    """(state, lam) that shortcut_fit.py chose for its bce head (best validation in-video AUC)."""
    fit = json.loads((RUN / "analysis-shortcut" / name / "fit.json").read_text())
    rows = [(fit["choices"][f"bce/{kind}/{s}"], s) for s in range(fit["n_states"])]
    choice, state = max(rows, key=lambda row: row[0]["validation"])
    return state, choice["setting"]["lam"]


def head_scores(fit_x: np.ndarray, fit_y: np.ndarray, score_x: np.ndarray, lam: float, device: str) -> np.ndarray:
    a, b = standardize(fit_x, score_x)
    head = fit_head(a, torch.tensor(fit_y), NO_PAIRS, 0.0, lam, device)
    with torch.no_grad():
        return head(b).squeeze(1).numpy()


def anchor_distance(x: np.ndarray, z_head: np.ndarray, previous: np.ndarray, prior: np.ndarray,
                    gamma: float) -> np.ndarray:
    """1 - cos(x_t, a_t) with the causal score-weighted anchor of the module docstring (x already standardized)."""
    out = np.empty(len(x))
    total = np.zeros_like(x)      # running sum of e_i x_i per row of the video chain
    weight = np.zeros(len(x))
    for t in range(len(x)):       # rows are in video order, windows in time order
        p = previous[t]
        if p >= 0:
            e = np.exp(-z_head[p] / gamma)
            total[t] = total[p] + e * x[p]
            weight[t] = weight[p] + e
        anchor = (K0 * prior + total[t]) / (K0 + weight[t])
        out[t] = 1 - x[t] @ anchor / (np.linalg.norm(x[t]) * np.linalg.norm(anchor) + 1e-8)
    return out


def zscore(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (values - ref.mean()) / (ref.std() + 1e-8)


def main(features: str, name: str, kind: str = "patch_mean", device: str = "cuda") -> None:
    features_dir = Path(features)
    out = RUN / "analysis-fusion" / f"{name}.json"
    if out.exists():
        raise FileExistsError(out)
    torch.cuda.set_per_process_memory_fraction(0.2)
    meta = read_meta(features_dir)
    train_meta = meta["train_anomaly"] + meta["train_normal"]
    videos = [row["video"] for row in train_meta]
    val = split(videos, {row["video"]: row["category"] for row in train_meta})
    is_val = np.array([video in val for video in videos])
    labels = np.array([row["holmes"] for row in train_meta], dtype=np.float32)
    normal = np.array([row["category"] == "Normal" for row in train_meta])
    val_meta = [row for row, keep in zip(train_meta, is_val, strict=True) if keep]
    val_y = labels[is_val] > 0.5

    state, lam = head_state(f"{name}-raw", kind)
    arrays = {s: np.asarray(np.load(features_dir / s / f"{kind}.npy", mmap_mode="r")[:, state], dtype=np.float32)
              for s in ("train_anomaly", "train_normal", "test")}
    train, test = np.concatenate([arrays["train_anomaly"], arrays["train_normal"]]), arrays["test"]
    head = {"val": head_scores(train[~is_val], labels[~is_val], train[is_val], lam, device),
            "test": head_scores(train, labels, test, lam, device)}
    column = FEATURE_NAMES.index("flow_mean")
    motion = np.log1p(np.concatenate([np.load(MOTION_DIR / s / "motion.npy") for s in ("train_anomaly", "train_normal")]))
    flow = {"val": motion[is_val, column], "test": np.log1p(np.load(MOTION_DIR / "test" / "motion.npy"))[:, column]}
    z_head = {k: zscore(v, head["val"]) for k, v in head.items()}
    z_flow = {k: zscore(v, flow["val"]) for k, v in flow.items()}

    # Anchor distance on features standardized with the train windows (fit split for validation).
    mean, std = train.mean(0), train.std(0) + 1e-6
    std_x = {"val": (train[is_val] - mean) / std, "test": (test - mean) / std}
    prior = ((train[normal & ~is_val] - mean) / std).mean(0)
    previous = {"val": previous_rows(val_meta), "test": previous_rows(meta["test"])}

    def pooled(values: np.ndarray) -> float:
        return float(roc_auc_score(val_y, values))

    chosen, curves = {}, {}
    chosen["head"] = (pooled(z_head["val"]), {}, z_head["test"])
    chosen["flow"] = (pooled(z_flow["val"]), {}, z_flow["test"])
    fusion = {w: {k: (1 - w) * z_head[k] + w * z_flow[k] for k in ("val", "test")} for w in WEIGHTS}
    best_w = max(WEIGHTS, key=lambda w: pooled(fusion[w]["val"]))
    chosen["fusion"] = (pooled(fusion[best_w]["val"]), {"w": best_w}, fusion[best_w]["test"])
    items = test_items()
    curves["fusion"] = {str(w): {"validation_pooled": pooled(fusion[w]["val"]),
                                 **evaluate(fusion[w]["test"], meta["test"], items)} for w in WEIGHTS}

    best = (-np.inf, None, None, None)
    for gamma in GAMMAS:
        dist = {k: anchor_distance(std_x[k], z_head[k], previous[k], prior, gamma) for k in ("val", "test")}
        z_dist = {k: zscore(v, dist["val"]) for k, v in dist.items()}
        if gamma == GAMMAS[0] or pooled(z_dist["val"]) > chosen["anchor_dist"][0]:
            chosen["anchor_dist"] = (pooled(z_dist["val"]), {"gamma": gamma}, z_dist["test"])
        for lam_a in LAMBDAS:
            combined = {k: z_head[k] + lam_a * z_dist[k] for k in ("val", "test")}
            value = pooled(combined["val"])
            if value > best[0]:
                best = (value, gamma, lam_a, combined["test"])
        print(f"gamma {gamma} done", flush=True)
    chosen["head_anchor"] = (best[0], {"gamma": best[1], "lam": best[2]}, best[3])

    results = {}
    for score, (value, setting, values) in chosen.items():
        results[score] = {"setting": setting, "validation_pooled": value, **evaluate(values, meta["test"], items)}
        print(line(f"{name}/{score} {setting} val {value:.3f}", results[score]), flush=True)
    for w in WEIGHTS:
        print(line(f"  fusion w {w} (test curve) val {curves['fusion'][str(w)]['validation_pooled']:.3f}",
                   curves["fusion"][str(w)]), flush=True)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"features": str(features_dir), "kind": kind, "state": state, "lam": lam,
                               "results": results, "fusion_curve": curves["fusion"]}, indent=2) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
