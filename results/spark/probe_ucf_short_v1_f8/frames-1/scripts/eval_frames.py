"""Frame AUC of the per-frame read-outs, paired with the frame-averaged features of vit-1 (2026-10-27).

fit-FV / fit-FM (one state): kind "last" = concat of 8 frame means, "visual" = mean and std over frames.
Baseline: vit-1 fit-V state 27 and fit-M state 0, kind "visual" (mean over all tokens of the 8 frames).
Same bootstrap as compare.json (N_BOOT video resamples, SEED). Run from the worktree root.
Writes ../eval_frames.json.
"""
import json
from pathlib import Path

import numpy as np

import s1stream.plugins  # noqa: F401  Register the dataset plugins.
from s1stream.probe.compare import N_BOOT, SEED, WINDOW_KEYS, _window_counts, _weighted_curve
from s1stream.probe.extract import load_probe_config
from s1stream.probe.fit import KINDS, METHODS
from s1stream.registry import DATASETS

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent
SOURCES = {"ViT 27": ("frames-1/fit-FV", "vit-1/fit-V", 27), "merger": ("frames-1/fit-FM", "vit-1/fit-M", 0)}
READS = {"concat": "last", "meanstd": "visual"}

config = load_probe_config(Path("configs/probe_ucf_short_v1_f8.yaml"))
items = {item.name: item for item in DATASETS.build(config.test).items()}
meta = [json.loads(line) for line in (ROOT / "run-2/fit-C/test_meta.jsonl").read_text().splitlines()]
for fit in ("frames-1/fit-FV", "frames-1/fit-FM"):
    other = [json.loads(line) for line in (ROOT / fit / "test_meta.jsonl").read_text().splitlines()]
    if [tuple(r[k] for k in WINDOW_KEYS) for r in other] != [tuple(r[k] for k in WINDOW_KEYS) for r in meta]:
        raise SystemExit(f"test windows of {fit} differ")
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
out = {"n_boot": N_BOOT, "seed": SEED, "p_yes_C": summary(p_yes), "results": {}}
for source, (fit, base_fit, state) in SOURCES.items():
    frames = np.load(ROOT / fit / "test_scores.npy")
    base = np.load(ROOT / base_fit / "test_scores.npy")
    for m, method in enumerate(METHODS):
        a_base = auc(base[m, KINDS.index("visual"), state])
        row = {"frame_average": summary(a_base)}
        for read, kind in READS.items():
            a = auc(frames[m, KINDS.index(kind), 0])
            row[read] = summary(a)
            row[f"{read}_minus_average"] = summary(a - a_base)
            row[f"{read}_minus_p_yes"] = summary(a - p_yes)
        out["results"][f"{source}/{method}"] = row
(RUN / "eval_frames.json").write_text(json.dumps(out, indent=1))
f = lambda s: f"{s['value']:.3f} [{s['low']:.3f}, {s['high']:.3f}]"
g = lambda s: f"{s['value']:+.3f} [{s['low']:+.3f}, {s['high']:+.3f}]"
for key, r in out["results"].items():
    print(f"{key:15s} avg {f(r['frame_average'])} | concat {f(r['concat'])} d {g(r['concat_minus_average'])} | "
          f"meanstd {f(r['meanstd'])} d {g(r['meanstd_minus_average'])}")
