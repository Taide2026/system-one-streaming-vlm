"""Frame AUC of the MLP probes, paired with the linear probe on the same features and with P(Yes) of C.

Same bootstrap as compare.json (N_BOOT video resamples, SEED). Linear scores: vit-1/fit-V, vit-1/fit-M, run-2/fit-C.
Run from the worktree root: uv run python results/spark/probe_ucf_short_v1_f8/mlp-1/scripts/eval_mlp.py
Writes ../eval_mlp.json.
"""
import json
from pathlib import Path

import numpy as np

import s1stream.plugins  # noqa: F401  Register the dataset plugins.
from s1stream.probe.compare import N_BOOT, SEED, _window_counts, _weighted_curve
from s1stream.probe.extract import load_probe_config
from s1stream.probe.fit import KINDS, METHODS
from s1stream.registry import DATASETS

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent
# target -> (linear fit folder, kind, state)
TARGETS = {"vit12": ("vit-1/fit-V", "visual", 12), "vit27": ("vit-1/fit-V", "visual", 27),
           "merger": ("vit-1/fit-M", "visual", 0), "lm_visual0": ("run-2/fit-C", "visual", 0),
           "lm_visual18": ("run-2/fit-C", "visual", 18), "lm_last23": ("run-2/fit-C", "last", 23)}

config = load_probe_config(Path("configs/probe_ucf_short_v1_f8.yaml"))
items = {item.name: item for item in DATASETS.build(config.test).items()}
meta = [json.loads(line) for line in (ROOT / "run-2/fit-C/test_meta.jsonl").read_text().splitlines()]
positive, negative = _window_counts(meta, items)
videos = list(dict.fromkeys(row["video"] for row in meta))
window_video = np.array([videos.index(row["video"]) for row in meta])
rng = np.random.default_rng(SEED)
draws = np.array([rng.integers(0, len(videos), len(videos)) for _ in range(N_BOOT)], dtype=int)
weights = np.zeros((N_BOOT + 1, len(videos)), dtype=np.int64)
weights[0] = 1
for i, draw in enumerate(draws, 1):
    weights[i] = np.bincount(draw, minlength=len(videos))
pos = weights[:, window_video] * positive
neg = weights[:, window_video] * negative


def auc(scores):
    return _weighted_curve(np.asarray(scores, dtype=np.float64), pos, neg)[0]


def summary(values):
    return {"value": round(float(values[0]), 4), "low": round(float(np.nanpercentile(values[1:], 2.5)), 4),
            "high": round(float(np.nanpercentile(values[1:], 97.5)), 4)}


p_yes = auc([row["p_yes"] for row in meta])
out = {"n_boot": N_BOOT, "seed": SEED, "p_yes_C": summary(p_yes), "targets": {}}
for name, (fit, kind, state) in TARGETS.items():
    mlp = np.load(RUN / "targets" / f"{name}.npy")
    linear = np.load(ROOT / fit / "test_scores.npy")
    row = {"features": fit, "kind": kind, "state": state}
    for i, method in enumerate(("mil", "holmes")):
        a_mlp = auc(mlp[i])
        a_lin = auc(linear[METHODS.index(method), KINDS.index(kind), state])
        row[method] = {"mlp": summary(a_mlp), "linear": summary(a_lin), "mlp_minus_linear": summary(a_mlp - a_lin),
                       "mlp_minus_p_yes": summary(a_mlp - p_yes)}
    out["targets"][name] = row
(RUN / "eval_mlp.json").write_text(json.dumps(out, indent=1))
print("p_yes", out["p_yes_C"])
for name, row in out["targets"].items():
    for method in ("mil", "holmes"):
        r = row[method]
        f = lambda s: f"{s['value']:.3f} [{s['low']:.3f}, {s['high']:.3f}]"
        g = lambda s: f"{s['value']:+.3f} [{s['low']:+.3f}, {s['high']:+.3f}]"
        print(f"{name:12s} {method:6s} linear {f(r['linear'])}  mlp {f(r['mlp'])}  mlp-lin {g(r['mlp_minus_linear'])}  mlp-pyes {g(r['mlp_minus_p_yes'])}")
