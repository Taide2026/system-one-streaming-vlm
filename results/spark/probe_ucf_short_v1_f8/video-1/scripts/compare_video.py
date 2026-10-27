"""Video input vs image input, per ViT state, paired (2026-10-27).

Inputs: ../fit-VV, ../fit-VM (video input) and ../../vit-1/fit-V, ../../vit-1/fit-M (image input).
Kinds: "visual" = token mean, "last" = token max. Same bootstrap as compare.json (N_BOOT, SEED).
Run from the worktree root. Writes ../compare_video.json and ../video_<method>.png.
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
ROOT = RUN.parent
POOL = {"visual": "mean", "last": "max"}
PAIRS = {"vit": ("video-1/fit-VV", "vit-1/fit-V"), "merger": ("video-1/fit-VM", "vit-1/fit-M")}

config = load_probe_config(Path("configs/probe_ucf_short_v1_f8.yaml"))
items = {item.name: item for item in DATASETS.build(config.test).items()}
meta = [json.loads(line) for line in (ROOT / "run-2/fit-C/test_meta.jsonl").read_text().splitlines()]
for video_fit, _ in PAIRS.values():
    other = [json.loads(line) for line in (ROOT / video_fit / "test_meta.jsonl").read_text().splitlines()]
    if [tuple(r[k] for k in WINDOW_KEYS) for r in other] != [tuple(r[k] for k in WINDOW_KEYS) for r in meta]:
        raise SystemExit(f"test windows of {video_fit} differ")
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


def curves(scores):
    return np.stack([_weighted_curve(np.asarray(s, dtype=np.float64), pos, neg)[0] for s in scores], axis=1)


def summary(v):
    return {"value": round(float(v[0]), 4), "low": round(float(np.nanpercentile(v[1:], 2.5)), 4),
            "high": round(float(np.nanpercentile(v[1:], 97.5)), 4)}


p_yes = _weighted_curve(np.asarray([r["p_yes"] for r in meta], dtype=np.float64), pos, neg)[0]
out = {"n_boot": N_BOOT, "seed": SEED, "p_yes_C": summary(p_yes), "curves": {}, "peaks": {}, "differences": {}}
plot = {}
for source, (video_fit, image_fit) in PAIRS.items():
    v_scores = np.load(ROOT / video_fit / "test_scores.npy")
    i_scores = np.load(ROOT / image_fit / "test_scores.npy")
    for m, method in enumerate(METHODS):
        for k, kind in enumerate(KINDS):
            vc, ic = curves(v_scores[m, k]), curves(i_scores[m, k])
            key = f"{source}/{method}/{POOL[kind]}"
            out["curves"][key] = {"video": vc[0].round(4).tolist(), "image": ic[0].round(4).tolist()}
            diff = vc - ic
            out["differences"][key] = [summary(diff[:, s]) for s in range(diff.shape[1])]
            vs, isx = int(np.argmax(vc[0])), int(np.argmax(ic[0]))
            out["peaks"][key] = {"video": {"state": vs, **summary(vc[:, vs])}, "image": {"state": isx, **summary(ic[:, isx])},
                                 "video_peak_minus_image_peak": summary(vc[:, vs] - ic[:, isx]),
                                 "video_peak_minus_p_yes": summary(vc[:, vs] - p_yes)}
            plot[key] = (vc, ic, diff)
(RUN / "compare_video.json").write_text(json.dumps(out, indent=1))

for method in METHODS:
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    vc, ic, diff = plot[f"vit/{method}/mean"]
    x = np.arange(vc.shape[1])
    ax.plot(x, ic[0], color="#2F5D9E", label="image input")
    ax.plot(x, vc[0], color="#B5651D", label="video input")
    ax.axhline(out["p_yes_C"]["value"], color="#14213D", ls="--", label="P(Yes) of C")
    ax.set_ylabel("test frame AUC")
    ax.set_title(f"{method}: ViT states, token mean")
    ax.legend(fontsize=8, loc="lower right")
    low, high = np.nanpercentile(diff[1:], [2.5, 97.5], axis=0)
    bx.plot(x, diff[0], color="#B5651D")
    bx.fill_between(x, low, high, color="#B5651D", alpha=0.2)
    bx.axhline(0, color="#4A5568", lw=1)
    bx.set_ylabel("video - image")
    bx.set_xlabel("ViT state (0 = patch + position embedding, s = after block s-1)")
    fig.tight_layout()
    fig.savefig(RUN / f"video_{method}.png", dpi=120)
    plt.close(fig)

for key, p in out["peaks"].items():
    f = lambda s: f"{s['value']:.3f} [{s['low']:.3f}, {s['high']:.3f}]"
    g = lambda s: f"{s['value']:+.3f} [{s['low']:+.3f}, {s['high']:+.3f}]"
    print(f"{key:20s} image @{p['image']['state']:2d} {f(p['image'])} | video @{p['video']['state']:2d} {f(p['video'])} | "
          f"peak diff {g(p['video_peak_minus_image_peak'])} | video - P(Yes) {g(p['video_peak_minus_p_yes'])}")
for key in ("vit/mil/mean", "vit/holmes/mean", "vit/knn/mean"):
    d = out["differences"][key][27]
    print(f"{key} state 27 video - image {d['value']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]")
