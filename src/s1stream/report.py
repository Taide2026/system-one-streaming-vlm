"""Offline stage: `s1stream report <run dir>` (proposal decisions 8 and 13). Never builds a backbone or a head.

report.json:
  windows, videos
  window_auroc, frame_auc, ece, brier     (None when the scored windows hold one label class only)
  seconds_per_window: {first, mean, median, p95}   (mean, median and p95 leave out the first window: warm-up)
  real_time_factor = seconds_per_window.mean / window.stride_s
  peak_gpu_gb, load_s                      (copied from run.json)
  rules: {<rule name>: [{setting, detection_rate, mean_delay_s, median_delay_s, false_alarms_per_hour}, ...]}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def report(run_dir: Path) -> dict[str, Any]:
    """Read config.yaml, manifest.json, scores.jsonl and run.json from run_dir, write run_dir / "report.json",
    and return the same dict.

    Only videos with at least one scored window count. VideoItem comes from the manifest; Window comes from each
    scores line (frame_indices left empty). Each decision rule of the config is built with DECISION_RULES and
    applied per video to that video's scores in window order; s1stream.metrics turns the alarms into the rows.
    """
    raise NotImplementedError
