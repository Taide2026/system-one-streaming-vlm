"""Split the frame AUC of the zero-shot and fine-tuned UCF-Crime runs on all 290 test videos (18,390 windows), as in
pyes_in_video_auc.py: all, vs_normal (event frames against normal-video frames), in_video (event frames against the
other frames of the anomaly videos) and the per-video in-video AUC. The last column is the in-video AUC on only the
49 anomaly videos of the 117-video short test set, the set of the probe numbers.
Zero-shot runs: results/pro6000/*_ucf_test_v1_f8 of the main checkout. Fine-tuned runs: results/pro6000/
ft_ucf_v1_*_eval/*/merged of the kv-across-windows worktree.
Usage: uv run python <this file>
"""

import json
import os
from pathlib import Path

import numpy as np

import s1stream.plugins  # noqa: F401  (registers the datasets)
from s1stream.config import load_config
from s1stream.encoder.extract import load_encoder_config
from s1stream.probe.compare import _weighted_curve, _window_counts
from s1stream.registry import DATASETS
from paths import ENCODER_RUN

MAIN = Path("/home/cctsai/Eric/VLM/system-one-streaming-vlm/results/pro6000")
FINETUNE = Path("/home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/kv-across-windows/results/pro6000")
ZERO_SHOT = ["qwen3_5_9b", "d3_flash", "clef_flash"]
FINETUNED = ["qwen", "d3flash", "clef", "clef_hard", "d3nano", "d3nano_full", "d3nano_hard", "d3nano_hardneg"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def latest(pattern: str, root: Path) -> Path:
    return sorted(root.glob(pattern))[-1]


def runs():
    """(label, scores.jsonl path) for each run."""
    for name in ZERO_SHOT:
        yield f"zero-shot {name}", latest(f"{name}_ucf_test_v1_f8/*/scores.jsonl", MAIN)
    for name in FINETUNED:
        yield f"fine-tune {name}", latest(f"ft_ucf_v1_{name}_eval/*/merged/scores.jsonl", FINETUNE)


def test_items() -> dict:
    """The 290 UCF-Crime test videos, from the zero-shot Qwen3.5-9B run config."""
    config_path = latest("qwen3_5_9b_ucf_test_v1_f8/*/config.yaml", MAIN)
    config = load_config(config_path, resolver=lambda asset: Path(os.environ["S1_ASSETS"]) / asset)
    return {item.name: item for item in DATASETS.build(config.dataset).items()}


def weighted_auc(values: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    return _weighted_curve(values, positive[None], negative[None])[0][0]


def per_video_aucs(values, positive, negative, videos, is_anomaly) -> np.ndarray:
    """In-video AUC of each anomaly video that has both event and other frames."""
    aucs = []
    for rows in videos:
        if not is_anomaly[rows[0]] or positive[rows].sum() == 0 or negative[rows].sum() == 0:
            continue
        aucs.append(weighted_auc(values[rows], positive[rows], negative[rows]))
    return np.array(aucs)


def main() -> None:
    items = test_items()
    short_test = {item.name for item in DATASETS.build(load_encoder_config(ENCODER_RUN / "config.yaml").test).items()}
    print(f"{'run':28s}  all    vs_normal in_video  per-video median  below 0.5  in_video short-49")
    for label, path in runs():
        meta = read_jsonl(path)
        values = np.array([row["score"] for row in meta], dtype=np.float64)
        positive, negative = _window_counts(meta, items)
        video_names = np.array([row["video"] for row in meta])
        is_anomaly = np.array([not video.startswith("Normal") for video in video_names])
        is_short = np.array([video in short_test for video in video_names])
        videos = [np.flatnonzero(video_names == video) for video in dict.fromkeys(video_names)]

        overall = weighted_auc(values, positive, negative)
        vs_normal = weighted_auc(values, positive, negative * ~is_anomaly)
        in_video = weighted_auc(values, positive, negative * is_anomaly)
        short_in_video = weighted_auc(values, positive * is_short, negative * (is_anomaly & is_short))
        per_video = per_video_aucs(values, positive, negative, videos, is_anomaly)
        below = int((per_video < 0.5).sum())
        print(f"{label:28s} {overall:.3f}  {vs_normal:.3f}     {in_video:.3f}     {np.median(per_video):.3f}"
              f"             {below:3d}/{len(per_video)}    {short_in_video:.3f}")


if __name__ == "__main__":
    main()
