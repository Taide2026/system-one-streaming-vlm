#!/usr/bin/env python3
"""The pilot's table: scripts/pilot-report.py <run dir> [<run dir> ...]

Per run (one context strategy each): the strategy, latency P95 and RTF, peak GPU memory, then per anomaly video the
p(abnormal) of its event windows (label 1) and whether the event is caught (any event window p >= 0.5; "2nd+" when a
window after the first one is), the number of events caught, and the false-positive rate at 0.5 over the normal
videos' windows. Reads scores.jsonl, config.yaml and run.json; report.json when present (else P95 from the scores).
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def describe(run: Path) -> None:
    config = yaml.safe_load((run / "config.yaml").read_text())
    rows = [json.loads(line) for line in (run / "scores.jsonl").read_text().splitlines()]
    run_data = json.loads((run / "run.json").read_text())
    context = config.get("context") or {"strategy": "s0"}
    label = " ".join(f"{k}={v}" for k, v in context.items())
    seconds = np.array([row["seconds"] for row in rows[1:]]) * 1000
    p95 = float(np.percentile(seconds, 95)) if seconds.size else float("nan")
    budget = config["window"]["stride_s"] * 1000
    peak = run_data.get("peak_gpu_gb")
    print(f"== {config['name']}: {label}")
    print(f"   {len(rows)} windows, latency P95 {p95:.0f} ms, RTF P95 {p95 / budget:.2f}, "
          f"peak GPU {peak:.1f} GB" if peak is not None else f"   {len(rows)} windows, latency P95 {p95:.0f} ms")
    by_video: dict[str, list[dict]] = {}
    for row in rows:
        by_video.setdefault(row["video"], []).append(row)
    caught = total = 0
    normal_windows: list[float] = []
    for video, video_rows in by_video.items():
        events = [row for row in video_rows if row["label"] == 1]
        if not events:
            normal_windows.extend(row["score"] for row in video_rows)
            continue
        total += 1
        hit = [row["score"] >= 0.5 for row in events]
        verdict = "caught 2nd+" if any(hit[1:]) else "caught 1st" if hit[0] else "missed"
        caught += any(hit)
        ps = ", ".join(f"w{row['window']} {row['score']:.2f}" for row in events)
        print(f"   {video:28s} {ps:40s} {verdict}")
    if normal_windows:
        fpr = sum(score >= 0.5 for score in normal_windows) / len(normal_windows)
        print(f"   events caught {caught}/{total}; FPR at 0.5 on {len(normal_windows)} normal windows: {fpr:.1%}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for run in sys.argv[1:]:
        describe(Path(run))


if __name__ == "__main__":
    main()
