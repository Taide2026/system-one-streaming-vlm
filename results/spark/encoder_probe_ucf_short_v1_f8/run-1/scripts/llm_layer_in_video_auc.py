"""In-video AUC of the LLM hidden-state probes of probe_ucf_short_v1_f8 run-2 (conditions A, B, C; kinds last and
visual), at the state chosen on validation (encoder.compare.select_state), split as in pyes_in_video_auc.py.
Same fit method (probe.fit) and same 117 test videos as the encoder probes, so the input feature is the only change.
Usage: uv run python <this file> [probe_ucf_short_v1_f8/run-2 folder of the main checkout]
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


def main(run: str = "/home/cctsai/Eric/VLM/system-one-streaming-vlm/results/spark/probe_ucf_short_v1_f8/run-2") -> None:
    items = {item.name: item for item in DATASETS.build(load_encoder_config(RUN / "config.yaml").test).items()}
    print(f"{'condition/kind/method':24s} state  all   vs_normal  in_video  per-video median  below 0.5")
    for fit_dir in sorted(Path(run).glob("fit-*")):
        fit = json.loads((fit_dir / "fit.json").read_text())
        scores = np.load(fit_dir / "test_scores.npy", mmap_mode="r")
        meta = [json.loads(line) for line in (fit_dir / "test_meta.jsonl").read_text().splitlines()]
        positive, negative = _window_counts(meta, items)
        video = np.array([row["video"] for row in meta])
        anomaly = np.array([row["category"] != "Normal" for row in meta])
        for kind in fit["kinds"]:
            for method in fit["methods"]:
                state = select_state(fit, method, kind)
                values = np.asarray(scores[fit["methods"].index(method), fit["kinds"].index(kind), state])

                def auc(pos: np.ndarray, neg: np.ndarray) -> float:
                    return _weighted_curve(values, pos[None], neg[None])[0][0]

                per_video = []
                for name in dict.fromkeys(video[anomaly]):
                    keep = video == name
                    if positive[keep].sum() and negative[keep].sum():
                        per_video.append(_weighted_curve(values[keep], positive[keep][None],
                                                         negative[keep][None])[0][0])
                per_video = np.array(per_video)
                print(f"{fit_dir.name.removeprefix('fit-')}/{kind}/{method}".ljust(24),
                      f"{state:5d}  {auc(positive, negative):.3f}  {auc(positive, negative * ~anomaly):.3f}      "
                      f"{auc(positive, negative * anomaly):.3f}     {np.median(per_video):.3f}             "
                      f"{int((per_video < 0.5).sum())}/{len(per_video)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
