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
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    run_data = json.loads((run_dir / "run.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "scores.jsonl").read_text().splitlines()]

    scores = [row["score"] for row in rows]
    labels = [row["label"] for row in rows]
    seconds = [row["seconds"] for row in rows]
    window = config["window"]
    thresholds = sorted({setting for spec in config["decision_rules"] if spec["name"] == "threshold"
                         for setting in DECISION_RULES.build(spec).settings()})
    if not thresholds:
        thresholds = [0.5]

    result = {
        "windows": len(rows),
        "videos": len({row["video"] for row in rows}),
        "frames_per_window": window["frames"],
        "peak_gpu_gb": run_data["peak_gpu_gb"],
        "load_s": run_data["load_s"],
        "latency": metrics.latency(seconds, window["frames"], window["stride_s"]),
        "classification": [metrics.classification(scores, labels, threshold) for threshold in thresholds],
    }
    (run_dir / "report.json").write_text(json.dumps(result, indent=1))
    return result
