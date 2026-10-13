"""Offline stage: `s1stream report <run dir>`. Never builds a backbone or a head.

report.json:
  windows: scored windows; videos: videos with at least one scored window
  frames_per_window: window.frames of the config
  peak_gpu_gb, load_s: copied from run.json
  latency: metrics.latency(seconds of every window in scores.jsonl order, window.frames, window.stride_s)
           = {first_window_ms, ms_per_frame, p50_ms, p95_ms, p99_ms, rtf_p95, budget_ms}
  classification: [metrics.classification(...) per threshold]  (window level; the settings of every "threshold"
                  rule in the config, sorted and without duplicates, or [0.5] when there is none)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

import s1stream.rules.threshold  # noqa: F401  Register the offline threshold rule.
from s1stream import metrics
from s1stream.registry import DECISION_RULES


def report(run_dir: Path) -> dict[str, Any]:
    """Read config.yaml, scores.jsonl and run.json from run_dir, write run_dir / "report.json" (json.dumps,
    indent=1), and return the same dict with exactly the keys in the module docstring.
    """
    raise NotImplementedError
