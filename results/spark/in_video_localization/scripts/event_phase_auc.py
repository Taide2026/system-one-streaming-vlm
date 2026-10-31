"""Which windows of the anomaly videos look abnormal: before, during or after the event?

Question: probes separate anomaly videos from normal videos (0.94-0.98) but not the event inside an anomaly video
(~0.6). If the windows before the event already score above the normal videos, the cue is the scene or the footage;
if only the windows after the event do, the cue is the aftermath (crowd, damage, police).
Each window of an anomaly test video gets one phase:
  during   it covers at least one annotated event frame
  before   it ends before the first event frame of its video
  after    it starts after the last event frame of its video
  between  it lies between two events of its video
For each phase: frame-weighted AUC of that phase's frames against all frames of the normal test videos
(0.5 = looks like normal footage; 1.0 = always scores above normal footage).
Scores (same 117 test videos): probe fits of encoder_probe_ucf_short_v1_f8/20261028-five-encoders at the state encoder-compare selects (validation only), and the
zero-shot p(yes) of d3-flash (results/spark/d3_flash_ucf_short_f8, main checkout).
Mode "full": the zero-shot and fine-tuned runs on all 290 test videos (the runs of finetune_in_video_auc.py).
Usage: uv run python <this file> [results/spark folder of the main checkout]
       uv run python <this file> full
Writes in_video_localization/05-event-phase/event-phase-auc.json (or event-phase-auc-full.json) and prints one line per score source.
"""

import json
import sys
from pathlib import Path

import numpy as np

from paths import ENCODER_RUN, EVENT_PHASE
import finetune_in_video_auc
import s1stream.plugins  # noqa: F401  (registers the datasets)
from s1stream.encoder.compare import select_state
from s1stream.encoder.extract import load_encoder_config
from s1stream.probe.compare import _weighted_curve, _window_counts
from s1stream.registry import DATASETS
from s1stream.windows import window_frame_range

PROBES = (("qwen35-vit", "patch_mean", "mil"), ("qwen35-vit", "patch_mean", "holmes"),
          ("vjepa2-vitl", "patch_mean", "mil"), ("vjepa2-vitl", "patch_mean", "holmes"))
PHASES = ("before", "during", "between", "after")


def probe_scores(fit_name: str, kind: str, method: str) -> tuple[list[dict], np.ndarray]:
    fit_dir = ENCODER_RUN / f"fit-{fit_name}"
    fit = json.loads((fit_dir / "fit.json").read_text())
    scores = np.load(fit_dir / "test_scores.npy", mmap_mode="r")
    meta = [json.loads(line) for line in (fit_dir / "test_meta.jsonl").read_text().splitlines()]
    state = select_state(fit, method, kind)
    return meta, np.asarray(scores[fit["methods"].index(method), fit["kinds"].index(kind), state], dtype=np.float64)


def pyes_scores(spark: Path) -> tuple[list[dict], np.ndarray]:
    run = sorted((spark / "d3_flash_ucf_short_f8").glob("*/scores.jsonl"))[-1]
    meta = [json.loads(line) for line in run.read_text().splitlines()]
    return meta, np.array([row["score"] for row in meta], dtype=np.float64)


def phase_of(row: dict, item, positive: int) -> str:
    """The phase of one anomaly-video window (see the module docstring)."""
    if positive > 0:
        return "during"
    first, last = window_frame_range(row["start_s"], row["end_s"], item.fps)
    event_start = min(start for start, _ in item.events)
    event_end = max(end for _, end in item.events)
    if last < event_start:
        return "before"
    if first > event_end:
        return "after"
    return "between"


def phase_auc(meta: list[dict], values: np.ndarray, items: dict) -> dict:
    positive, negative = _window_counts(meta, items)
    frames = positive + negative
    normal = np.array([row["video"].startswith("Normal") for row in meta])
    phase = np.array(["normal" if normal[i] else phase_of(row, items[row["video"]], positive[i])
                      for i, row in enumerate(meta)])
    out = {}
    for name in PHASES:
        keep = phase == name
        if not keep.any():
            continue
        auc = _weighted_curve(values, (frames * keep)[None], (frames * normal)[None])[0][0]
        videos = {meta[i]["video"] for i in np.flatnonzero(keep)}
        out[name] = {"auc_vs_normal": float(auc), "windows": int(keep.sum()), "videos": len(videos)}
    return out


def report(sources: dict, items: dict, out_name: str) -> None:
    results = {}
    print(f"{'source':32s} " + " ".join(f"{name:>16s}" for name in PHASES) + "   (AUC vs normal, windows)")
    for name, (meta, values) in sources.items():
        results[name] = phase_auc(meta, values, items)
        cells = [f"{results[name][p]['auc_vs_normal']:.3f} ({results[name][p]['windows']:4d})" if p in results[name]
                 else f"{'-':>16s}" for p in PHASES]
        print(f"{name:32s} " + " ".join(f"{cell:>16s}" for cell in cells))
    out = EVENT_PHASE
    out.mkdir(exist_ok=True)
    (out / out_name).write_text(json.dumps(results, indent=2) + "\n")


def main_full() -> None:
    """Zero-shot and fine-tuned runs on the 290 test videos."""
    items = finetune_in_video_auc.test_items()
    sources = {}
    for label, path in finetune_in_video_auc.runs():
        meta = finetune_in_video_auc.read_jsonl(path)
        sources[label] = (meta, np.array([row["score"] for row in meta], dtype=np.float64))
    report(sources, items, "event-phase-auc-full.json")


def main(spark: str = "/home/cctsai/Eric/VLM/system-one-streaming-vlm/results/spark") -> None:
    items = {item.name: item for item in DATASETS.build(load_encoder_config(ENCODER_RUN / "config.yaml").test).items()}
    sources = {f"{fit}/{kind}/{method}": probe_scores(fit, kind, method) for fit, kind, method in PROBES}
    sources["d3-flash/p_yes"] = pyes_scores(Path(spark))
    report(sources, items, "event-phase-auc.json")


if __name__ == "__main__":
    if sys.argv[1:] == ["full"]:
        main_full()
    else:
        main(*sys.argv[1:])
