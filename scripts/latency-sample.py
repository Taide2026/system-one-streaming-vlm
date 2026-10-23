#!/usr/bin/env python3
"""Latency re-measurement by sampling (2026-10-24): the first N windows of every test video, on a GPU whose venv has
the fast Gated DeltaNet kernels (flash-linear-attention), for one of main's configs.

  PYTHONPATH=<main>/src python scripts/latency-sample.py <config.yaml> --out <dir> [--per-video 2] [--warmup 5]
                                                          [--compare <full run dir>]

Why: every pro6000 / spark run so far used a venv without flash-linear-attention, so the Qwen3.5-based models (d3,
Clef, StartLux, H2O, Qwen3.5-9B) ran the reference DeltaNet implementation: the same probabilities, a higher latency.
The accuracy runs stand; this script re-measures the latency only. Sampling the first N windows of every video keeps
the mix of resolutions and codecs of the full run; the first `warmup` windows are run but left out of the statistics.

Runs against the code the config belongs to (main: all backbones), so put that checkout's src on PYTHONPATH. Writes
scores.jsonl (one line per window, as the runner), run.json and latency.json {windows, p50_ms, p95_ms, p99_ms,
ms_per_frame, rtf_p95, kernels, compare: {full run's latency, delta}} into --out.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
import yaml

import s1stream.plugins  # noqa: F401
from s1stream import metrics
from s1stream.config import load_config
from s1stream.registry import ANSWER_HEADS, BACKBONES, DATASETS, INPUT_FORMS
from s1stream.windows import plan_windows


def kernels() -> dict[str, bool]:
    out = {}
    for name in ("fla", "causal_conv1d"):
        try:
            __import__(name)
            out[name] = True
        except ImportError:
            out[name] = False
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-video", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.yaml").write_text(yaml.safe_dump(config.raw, sort_keys=False))
    items = DATASETS.build(config.dataset).items()
    input_form = INPUT_FORMS.build(config.input_form)
    backbone = BACKBONES.build(config.backbone)
    head = ANSWER_HEADS.build(config.answer_head)
    cuda = torch.cuda.is_available()
    seconds: list[float] = []
    rows = []
    with (args.out / "scores.jsonl").open("w") as stream:
        for item in items:
            for window in plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames)[:args.per_video]:
                if cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    inp = input_form.build(item, window, config.question)
                    probabilities = head.probabilities(backbone.forward(inp, config.question), config.question)
                if cuda:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                row = {"video": item.name, "window": window.index, "label": window.label,
                       "score": config.question.score(probabilities), "seconds": elapsed}
                stream.write(json.dumps(row) + "\n")
                rows.append(row)
                seconds.append(elapsed)
    measured = seconds[args.warmup:]
    latency = metrics.latency(measured, config.window.frames, config.window.stride_s)
    result = {"windows": len(measured), "warmup": args.warmup, "per_video": args.per_video, **latency,
              "kernels": kernels(), "gpu": torch.cuda.get_device_name() if cuda else None,
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "transformers": version("transformers")},
              "finished": datetime.now(timezone.utc).isoformat()}
    if args.compare:
        full = json.loads((args.compare / "report.json").read_text())["latency"]
        result["compare"] = {"full_run": str(args.compare), "full": full,
                             "delta_p95_ms": latency["p95_ms"] - full["p95_ms"],
                             "ratio_p95": latency["p95_ms"] / full["p95_ms"] if full["p95_ms"] else None}
    (args.out / "latency.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
