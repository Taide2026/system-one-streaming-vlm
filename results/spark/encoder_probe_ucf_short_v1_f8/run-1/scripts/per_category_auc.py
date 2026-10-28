"""Test frame AUC per UCF-Crime category, at the state that encoder-compare selects (validation only).

Each category AUC uses all frames of that category's test videos plus all frames of the normal test videos,
weighted as in encoder-compare. "in_video" uses only the frames of the anomaly videos (event frames against the
other frames of the same kind of videos), so it measures localization in time, not video-level separation; (probe.compare._window_counts, _weighted_curve).
Usage: uv run python <this file> [method]   (method: mil, holmes or knn; default mil)
"""

import json
import sys
from pathlib import Path

import numpy as np

import s1stream.plugins  # noqa: F401  (registers the datasets)

from s1stream.encoder.compare import select_state
from s1stream.encoder.extract import load_encoder_config
from s1stream.probe.compare import _weighted_curve, _window_counts
from s1stream.registry import DATASETS

RUN = Path(__file__).resolve().parents[1]


def main(method: str = "mil") -> None:
    items = {item.name: item for item in DATASETS.build(load_encoder_config(RUN / "config.yaml").test).items()}
    rows = {}
    for fit_dir in sorted(RUN.glob("fit-*")):
        fit = json.loads((fit_dir / "fit.json").read_text())
        scores = np.load(fit_dir / "test_scores.npy", mmap_mode="r")
        meta = [json.loads(line) for line in (fit_dir / "test_meta.jsonl").read_text().splitlines()]
        positive, negative = _window_counts(meta, items)
        category = np.array([row["category"] for row in meta])
        categories = sorted(set(category) - {"Normal"})
        for kind in fit["kinds"]:
            state = select_state(fit, method, kind)
            values = np.asarray(scores[fit["methods"].index(method), fit["kinds"].index(kind), state])
            out = {"state": state, "all": _weighted_curve(values, positive[None], negative[None])[0][0]}
            anomaly = category != "Normal"
            out["in_video"] = _weighted_curve(values, (positive * anomaly)[None], (negative * anomaly)[None])[0][0]
            for name in categories:
                keep = (category == name) | (category == "Normal")
                out[name] = _weighted_curve(values, (positive * keep)[None], (negative * keep)[None])[0][0]
            rows[f"{fit_dir.name.removeprefix('fit-')}/{kind}"] = out
    videos = {name: len({row["video"] for row in meta if row["category"] == name}) for name in categories}
    print("test videos:", videos)
    print(f"{'encoder/kind':32s} st   all  in_vid " + " ".join(f"{name[:6]:>6s}" for name in categories))
    for key, out in rows.items():
        print(f"{key:32s} {out['state']:2d} {out['all']:.3f} {out['in_video']:.3f}  " + " ".join(f"{out[name]:6.3f}" for name in categories))


if __name__ == "__main__":
    main(*sys.argv[1:])
