"""Offline stage: `s1stream report <run dir>` (proposal decisions 8 and 13). Never builds a backbone or a head.

report.json:
  windows, videos
  window_auroc, frame_auc, ece, brier
  seconds_per_window: {first, mean, median, p95}   (mean, median and p95 leave out the first window: warm-up)
  real_time_factor = seconds_per_window.mean / window.stride_s
  peak_gpu_gb, load_s                      (copied from run.json)
  rules: {<rule name>: [{setting, detection_rate, mean_delay_s, median_delay_s, false_alarms_per_hour}, ...]}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import s1stream.rules.threshold  # noqa: F401  Register the offline threshold rule.
from s1stream import metrics
from s1stream.registry import DECISION_RULES
from s1stream.types import VideoItem, Window


def report(run_dir: Path) -> dict[str, Any]:
    """Read config.yaml, manifest.json, scores.jsonl and run.json from run_dir, write run_dir / "report.json",
    and return the same dict.

    Only videos with at least one scored window count. VideoItem comes from the manifest; Window comes from each
    scores line (frame_indices left empty). Each decision rule of the config is built with DECISION_RULES and
    applied per video to that video's scores in window order; s1stream.metrics turns the alarms into the rows.
    """
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    run_data = json.loads((run_dir / "run.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "scores.jsonl").read_text().splitlines()]
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[row["video"]].append(row)

    videos = []
    for entry in manifest:
        video_rows = by_video.get(entry["name"])
        if not video_rows:
            continue
        item = VideoItem(entry["name"], Path(entry["path"]), entry["category"], entry["fps"],
                         entry["n_frames"], tuple(tuple(event) for event in entry["events"]))
        video_rows.sort(key=lambda row: row["window"])
        windows = [Window(item.name, row["window"], row["start_s"], row["end_s"], (), row["label"])
                   for row in video_rows]
        scores = [row["score"] for row in video_rows]
        videos.append((item, windows, scores))

    scores = [row["score"] for row in rows]
    labels = [row["label"] for row in rows]
    both_classes = len(set(labels)) == 2
    times = [row["seconds"] for row in rows]
    after_warmup = times[1:]
    timing = {"first": times[0] if times else None,
              "mean": float(np.mean(after_warmup)) if after_warmup else None,
              "median": float(np.median(after_warmup)) if after_warmup else None,
              "p95": float(np.percentile(after_warmup, 95)) if after_warmup else None}
    stride = config["window"]["stride_s"]
    rules = {}
    for spec in config["decision_rules"]:
        rule = DECISION_RULES.build(spec)
        settings = rule.settings()
        alarm_sets = [rule.alarms(video_scores) for _, _, video_scores in videos]
        rule_rows = []
        for index, setting in enumerate(settings):
            results = [metrics.video_alarms(item, windows, alarms[index])
                       for (item, windows, _), alarms in zip(videos, alarm_sets, strict=True)]
            rule_rows.append({"setting": setting, **metrics.summarize_alarms(results)})
        rules[spec["name"]] = rule_rows

    result = {"windows": len(rows), "videos": len(videos),
              "window_auroc": metrics.auroc(scores, labels) if both_classes else None,
              "frame_auc": metrics.frame_auc(videos) if both_classes else None,
              "ece": metrics.ece(scores, labels), "brier": metrics.brier(scores, labels) if rows else None,
              "seconds_per_window": timing,
              "real_time_factor": timing["mean"] / stride if timing["mean"] is not None else None,
              "peak_gpu_gb": run_data["peak_gpu_gb"], "load_s": run_data["load_s"], "rules": rules}
    (run_dir / "report.json").write_text(json.dumps(result, indent=1))
    return result
