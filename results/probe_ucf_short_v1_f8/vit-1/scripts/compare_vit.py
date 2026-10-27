"""Frame AUC per ViT state, and paired differences against the LM probes of run-2 (2026-10-26 spike).

Inputs: ../fit-V and ../fit-M (probe-fit on the ViT and merger features of extract_vit.py), ../../run-2/fit-C.
Same bootstrap as compare.json (compare.N_BOOT video resamples, compare.SEED), so every difference is paired.
Kinds of fit-V and fit-M: "visual" = token mean, "last" = token max (see extract_vit.py).
Peak states are chosen on the test set (as in the deck), so peak values are slightly optimistic.

Run from the worktree root: uv run python results/probe_ucf_short_v1_f8/vit-1/scripts/compare_vit.py
Writes ../compare_vit.json and ../vit_<method>.png.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import s1stream.plugins  # noqa: F401  Register the dataset plugins.
from s1stream.probe.compare import N_BOOT, SEED, WINDOW_KEYS, _window_counts, _weighted_curve
from s1stream.probe.extract import load_probe_config
from s1stream.probe.fit import KINDS, METHODS
from s1stream.registry import DATASETS

RUN = Path(__file__).resolve().parents[1]
RUN2 = RUN.parent / "run-2"
POOL = {"visual": "mean", "last": "max"}


def load(fit: Path) -> tuple[np.ndarray, list[dict]]:
    meta = [json.loads(line) for line in (fit / "test_meta.jsonl").read_text().splitlines()]
    return np.load(fit / "test_scores.npy"), meta


def interval(samples: np.ndarray) -> tuple[float, float]:
    return float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))


config = load_probe_config(Path("configs/probe_ucf_short_v1_f8.yaml"))
items = {item.name: item for item in DATASETS.build(config.test).items()}
fits = {name: load(path) for name, path in (("V", RUN / "fit-V"), ("M", RUN / "fit-M"), ("C", RUN2 / "fit-C"))}
meta = fits["C"][1]
for name, (_, other) in fits.items():
    if [tuple(row[k] for k in WINDOW_KEYS) for row in other] != [tuple(row[k] for k in WINDOW_KEYS) for row in meta]:
        raise SystemExit(f"test windows of fit-{name} differ from fit-C")

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


def curves(scores: np.ndarray) -> np.ndarray:
    """(N_BOOT + 1, S) frame AUC; row 0 = original test set."""
    return np.stack([_weighted_curve(s, pos, neg)[0] for s in scores], axis=1)


p_yes = _weighted_curve(np.asarray([row["p_yes"] for row in meta]), pos, neg)[0]
out = {"n_boot": N_BOOT, "seed": SEED, "kinds": POOL,
       "p_yes_C": {"value": float(p_yes[0]), **dict(zip(("low", "high"), interval(p_yes[1:])))},
       "curves": {}, "peaks": {}, "differences": {}}
all_curves = {}
for method_i, method in enumerate(METHODS):
    lm_visual = curves(fits["C"][0][method_i, KINDS.index("visual")])
    lm_last = curves(fits["C"][0][method_i, KINDS.index("last")])
    refs = {"C_visual_state0": lm_visual[:, 0], "C_visual_peak": lm_visual[:, int(np.argmax(lm_visual[0]))],
            "C_last_peak": lm_last[:, int(np.argmax(lm_last[0]))], "p_yes_C": p_yes}
    for kind_i, kind in enumerate(KINDS):
        for source in ("V", "M"):
            curve = curves(fits[source][0][method_i, kind_i])
            key = f"{source}/{method}/{POOL[kind]}"
            all_curves[key] = curve
            low, high = np.nanpercentile(curve[1:], [2.5, 97.5], axis=0)
            out["curves"][key] = {"value": curve[0].round(4).tolist(), "low": low.round(4).tolist(),
                                  "high": high.round(4).tolist()}
            state = int(np.argmax(curve[0]))
            out["peaks"][key] = {"state": state, "value": float(curve[0, state]),
                                 **dict(zip(("low", "high"), interval(curve[1:, state])))}
            for ref, ref_curve in refs.items():
                diff = curve[:, state] - ref_curve
                out["differences"][f"{key} - {ref}"] = {"value": float(diff[0]),
                                                       **dict(zip(("low", "high"), interval(diff[1:])))}
    for ref in ("C_visual_state0", "C_visual_peak", "C_last_peak"):
        out["peaks"][f"{ref}/{method}"] = {"value": float(refs[ref][0]), **dict(zip(("low", "high"),
                                                                                interval(refs[ref][1:])))}
(RUN / "compare_vit.json").write_text(json.dumps(out, indent=1))

for method in METHODS:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind, color in (("mean", "#2F5D9E"), ("max", "#B5651D")):
        c = out["curves"][f"V/{method}/{kind}"]
        x = np.arange(len(c["value"]))
        ax.plot(x, c["value"], color=color, label=f"ViT token {kind}")
        ax.fill_between(x, c["low"], c["high"], color=color, alpha=0.15)
        m = out["peaks"][f"M/{method}/{kind}"]
        ax.errorbar([len(x)], [m["value"]], yerr=[[m["value"] - m["low"]], [m["high"] - m["value"]]], color=color,
                    marker="s", capsize=3)
    ax.axhline(out["peaks"][f"C_visual_state0/{method}"]["value"], color="#4A5568", ls=":", label="LM state 0 visual mean (C)")
    ax.axhline(out["peaks"][f"C_last_peak/{method}"]["value"], color="#4A5568", ls="-.", label="LM best last token (C)")
    ax.axhline(out["p_yes_C"]["value"], color="#14213D", ls="--", label="P(Yes) of C")
    ax.set_xticks(list(range(0, len(x), 3)) + [len(x)], [str(i) for i in range(0, len(x), 3)] + ["merger"])
    ax.set_xlabel("ViT state (0 = patch + position embedding, s = after block s-1)")
    ax.set_ylabel("test frame AUC")
    ax.set_title(f"{method}: ViT states, 95 % video-bootstrap band")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RUN / f"vit_{method}.png", dpi=120)
    plt.close(fig)

for key, peak in out["peaks"].items():
    state = f"@{peak['state']:2d}" if "state" in peak else "   "
    print(f"{key:26s} {state} {peak['value']:.3f} [{peak['low']:.3f}, {peak['high']:.3f}]")
for key, diff in out["differences"].items():
    print(f"{key:44s} {diff['value']:+.3f} [{diff['low']:+.3f}, {diff['high']:+.3f}]")
