"""Split the test frame AUC of the zero-shot p(yes) runs (results/spark/*_ucf_short_f8, same 117 test videos) by the
source of the negative frames, as for the probes: "vs_normal" = event frames against the frames of the normal videos,
"in_video" = event frames against the non-event frames of the anomaly videos, and the median per-video in-video AUC.
Usage: uv run python <this file> [results/spark folder of the main checkout]
"""

import json
import sys
from pathlib import Path

import numpy as np

import s1stream.plugins  # noqa: F401  (registers the datasets)
from s1stream.encoder.extract import load_encoder_config
from s1stream.probe.compare import _weighted_curve, _window_counts
from s1stream.registry import DATASETS
from paths import ENCODER_RUN



def main(spark: str = "/home/cctsai/Eric/VLM/system-one-streaming-vlm/results/spark") -> None:
    items = {item.name: item for item in DATASETS.build(load_encoder_config(ENCODER_RUN / "config.yaml").test).items()}
    print(f"{'run':46s}  all   vs_normal  in_video  per-video median  below 0.5")
    for folder in sorted(Path(spark).glob("*_ucf_short_f8")):
        runs = sorted(folder.glob("*/scores.jsonl"))
        if not runs:
            continue
        meta = [json.loads(line) for line in runs[-1].read_text().splitlines()]
        positive, negative = _window_counts(meta, items)
        values = np.array([row["score"] for row in meta], dtype=np.float64)
        video = np.array([row["video"] for row in meta])
        anomaly = np.array([not name.startswith("Normal") for name in video])

        def auc(pos: np.ndarray, neg: np.ndarray) -> float:
            return _weighted_curve(values, pos[None], neg[None])[0][0]

        per_video = []
        for name in dict.fromkeys(video[anomaly]):
            keep = video == name
            if positive[keep].sum() and negative[keep].sum():
                per_video.append(_weighted_curve(values[keep], positive[keep][None], negative[keep][None])[0][0])
        per_video = np.array(per_video)
        print(f"{folder.name}/{runs[-1].parent.name:15s}"[:46].ljust(46),
              f"{auc(positive, negative):.3f}  {auc(positive, negative * ~anomaly):.3f}      "
              f"{auc(positive, negative * anomaly):.3f}     {np.median(per_video):.3f}             "
              f"{int((per_video < 0.5).sum())}/{len(per_video)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
