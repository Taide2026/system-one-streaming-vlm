"""In-video frame AUC per anomaly test video (event frames against the other frames of the same video), at the state
that encoder-compare selects. Videos whose frames are all event or all normal have no AUC and are left out.
Usage: uv run python <this file> [fit] [kind] [method]   (default qwen35-vit patch_mean mil)
Prints one line per video, sorted by AUC, and writes per_video_<fit>_<kind>_<method>.json with the window scores.
"""

import json
import sys

import numpy as np

import s1stream.plugins  # noqa: F401  (registers the datasets)
from s1stream.encoder.compare import select_state
from s1stream.encoder.extract import load_encoder_config
from s1stream.probe.compare import _weighted_curve, _window_counts
from s1stream.registry import DATASETS
from paths import ENCODER_RUN, FRAME_SPLIT



def main(fit_name: str = "qwen35-vit", kind: str = "patch_mean", method: str = "mil") -> None:
    items = {item.name: item for item in DATASETS.build(load_encoder_config(ENCODER_RUN / "config.yaml").test).items()}
    fit_dir = ENCODER_RUN / f"fit-{fit_name}"
    fit = json.loads((fit_dir / "fit.json").read_text())
    scores = np.load(fit_dir / "test_scores.npy", mmap_mode="r")
    meta = [json.loads(line) for line in (fit_dir / "test_meta.jsonl").read_text().splitlines()]
    positive, negative = _window_counts(meta, items)
    state = select_state(fit, method, kind)
    values = np.asarray(scores[fit["methods"].index(method), fit["kinds"].index(kind), state])
    video = np.array([row["video"] for row in meta])
    out = {}
    for name in dict.fromkeys(video):
        keep = video == name
        if meta[int(np.flatnonzero(keep)[0])]["category"] == "Normal":
            continue
        if positive[keep].sum() == 0 or negative[keep].sum() == 0:
            continue
        auc = _weighted_curve(values[keep], positive[keep][None], negative[keep][None])[0][0]
        rows = np.flatnonzero(keep)
        out[name] = {"auc": float(auc), "category": meta[rows[0]]["category"],
                     "windows": [{"start_s": meta[i]["start_s"], "end_s": meta[i]["end_s"],
                                  "label": meta[i]["label"], "score": float(values[i])} for i in rows]}
    ranked = sorted(out, key=lambda name: out[name]["auc"])
    print(f"{fit_name}/{kind}/{method} state {state}: {len(out)} videos")
    for name in ranked:
        w = out[name]["windows"]
        print(f"{out[name]['auc']:.3f}  {name:28s} windows {len(w):3d}  event {sum(x['label'] for x in w):3d}")
    path = FRAME_SPLIT / f"per_video_{fit_name}_{kind}_{method}.json"
    path.write_text(json.dumps({"state": state, "videos": out}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
