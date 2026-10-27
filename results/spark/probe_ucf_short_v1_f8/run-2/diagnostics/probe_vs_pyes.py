"""Paired video bootstrap of (probe frame AUC - P(Yes) frame AUC) for condition C (2026-10-26).

Uses the same resamples as compare.json (compare.bootstrap: N_BOOT, SEED, video draws), so the probe AUC and the
P(Yes) AUC are computed on the same videos in every resample. The probe state is the test peak state of each curve
(as reported in the deck), so the probe side is slightly optimistic. Only C: the meta "p_yes" of A and B still reads
the "Yes"/"No" tokens (see ../README.md).

Run from the worktree root: uv run python results/spark/probe_ucf_short_v1_f8/run-2/diagnostics/probe_vs_pyes.py
Writes probe_vs_pyes.json next to this script.
"""
import json
from pathlib import Path

import numpy as np

import s1stream.plugins  # noqa: F401  Register the dataset plugins.
from s1stream.probe.compare import N_BOOT, SEED, _window_counts, _weighted_curve
from s1stream.probe.extract import load_probe_config
from s1stream.probe.fit import KINDS, METHODS
from s1stream.registry import DATASETS

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
config = load_probe_config(Path("configs/probe_ucf_short_v1_f8.yaml"))
items = {item.name: item for item in DATASETS.build(config.test).items()}
meta = [json.loads(line) for line in (RUN / "fit-C" / "test_meta.jsonl").read_text().splitlines()]
scores = np.load(RUN / "fit-C" / "test_scores.npy")
p_yes = np.asarray([row["p_yes"] for row in meta])

positive, negative = _window_counts(meta, items)
videos = list(dict.fromkeys(row["video"] for row in meta))
video_index = {name: i for i, name in enumerate(videos)}
window_video = np.array([video_index[row["video"]] for row in meta])
rng = np.random.default_rng(SEED)
draws = np.array([rng.integers(0, len(videos), len(videos)) for _ in range(N_BOOT)], dtype=int)
weights = np.zeros((N_BOOT + 1, len(videos)), dtype=np.int64)
weights[0] = 1
for i, draw in enumerate(draws, 1):
    weights[i] = np.bincount(draw, minlength=len(videos))
pos = weights[:, window_video] * positive
neg = weights[:, window_video] * negative

pyes_auc, _ = _weighted_curve(p_yes, pos, neg)
out = {"n_boot": N_BOOT, "seed": SEED, "n_videos": len(videos),
       "p_yes": {"value": float(pyes_auc[0]), "low": float(np.nanpercentile(pyes_auc[1:], 2.5)),
                 "high": float(np.nanpercentile(pyes_auc[1:], 97.5))},
       "probes": {}}
for method_i, method in enumerate(METHODS):
    for kind_i, kind in enumerate(KINDS):
        curves = np.stack([_weighted_curve(s, pos, neg)[0] for s in scores[method_i, kind_i]], axis=1)
        state = int(np.argmax(curves[0]))
        diff = curves[:, state] - pyes_auc
        samples = diff[1:]
        out["probes"][f"{method}/{kind}"] = {
            "state": state, "probe": float(curves[0, state]), "diff": float(diff[0]),
            "low": float(np.nanpercentile(samples, 2.5)), "high": float(np.nanpercentile(samples, 97.5)),
            "share_probe_better": float(np.nanmean(samples > 0))}
        if (method, kind) == ("holmes", "last"):
            out["holmes_last_diff_samples"] = [round(float(x), 5) for x in samples]
(HERE / "probe_vs_pyes.json").write_text(json.dumps(out, indent=1))
for key, row in out["probes"].items():
    print(f"{key:14s} state {row['state']:2d} probe {row['probe']:.3f} diff {row['diff']:+.3f} "
          f"[{row['low']:+.3f}, {row['high']:+.3f}] P(diff>0) {row['share_probe_better']:.2f}")
print("p_yes", out["p_yes"])
