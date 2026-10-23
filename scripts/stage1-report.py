#!/usr/bin/env python3
"""Stage 1 table: scripts/stage1-report.py <zero-shot run dir> <fine-tuned run dir> [<training run dir>]

Both runs score the same UCF-Crime test windows (configs zs_ucf_v1_*_eval / ft_ucf_v1_*_eval). Per run: frame AUC and
AP (report.json, or computed here when report.json is missing), recall at 5 % frame FPR, window-level 10-bin ECE of
p(abnormal) against the window label, FPR at 0.5 on normal videos, latency P95 and RTF. With the training run dir:
its steps, wall time, peak GPU memory and the calibration fitted on held-out training videos. Prints a Typst table
body too (for the deck).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1stream import metrics  # noqa: E402
from s1stream.report import frame_videos  # noqa: E402


def load(run: Path) -> dict:
    rows = [json.loads(line) for line in (run / "scores.jsonl").read_text().splitlines()]
    manifest = {video["name"]: video for video in json.loads((run / "manifest.json").read_text())}
    report_file = run / "report.json"
    report = json.loads(report_file.read_text()) if report_file.is_file() else None
    frames = metrics.frame_scores(frame_videos(rows, manifest))
    scores, labels = frames
    order = np.argsort(-scores)
    pos, neg = labels.sum(), len(labels) - labels.sum()
    tp = np.cumsum(labels[order]) / pos
    fp = np.cumsum(1 - labels[order]) / neg
    recall_at_5 = float(tp[fp <= 0.05].max()) if np.any(fp <= 0.05) else 0.0
    window_scores = np.array([row["score"] for row in rows])
    window_labels = np.array([row["label"] for row in rows])
    bins = np.minimum((window_scores * 10).astype(int), 9)
    ece = sum(abs(window_labels[bins == b].mean() - window_scores[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if np.any(bins == b))
    normal = np.array([row["score"] for row in rows if not manifest[row["video"]]["events"]])
    seconds = [row["seconds"] for row in rows]
    latency = report["latency"] if report else metrics.latency(seconds, 8, 2.0)
    return {"windows": len(rows), "frame_auc": metrics.auc(scores, labels), "frame_ap": metrics.ap(scores, labels),
            "recall_fpr5": recall_at_5, "ece": float(ece), "fpr_normal_0.5": float((normal >= 0.5).mean()),
            "p95_ms": latency["p95_ms"], "rtf": latency["rtf_p95"]}


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    zs, ft = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    print(f"{'':22s}{'zero-shot':>12s}{'fine-tuned':>12s}")
    for key in ("windows", "frame_auc", "frame_ap", "recall_fpr5", "ece", "fpr_normal_0.5", "p95_ms", "rtf"):
        a, b = zs[key], ft[key]
        fmt = (lambda v: f"{v:12.4f}") if isinstance(a, float) else (lambda v: f"{v:12.0f}")
        print(f"{key:22s}{fmt(a)}{fmt(b)}")
    if len(sys.argv) > 3:
        run = json.loads((Path(sys.argv[3]) / "run.json").read_text())
        print(json.dumps({k: run[k] for k in ("steps", "train_windows", "train_seconds", "peak_gpu_gb", "calibration")},
                         indent=1))
    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    print("\n// Typst rows: zero-shot / fine-tuned")
    print(f"[zero-shot], [{pct(zs['frame_auc'])}], [{pct(zs['frame_ap'])}], [{pct(zs['recall_fpr5'])}], "
          f"[{zs['ece']:.3f}], [{pct(zs['fpr_normal_0.5'])}], [{zs['p95_ms']:.0f} ms], [{zs['rtf']:.2f}],")
    print(f"[fine-tuned], [{pct(ft['frame_auc'])}], [{pct(ft['frame_ap'])}], [{pct(ft['recall_fpr5'])}], "
          f"[{ft['ece']:.3f}], [{pct(ft['fpr_normal_0.5'])}], [{ft['p95_ms']:.0f} ms], [{ft['rtf']:.2f}],")


if __name__ == "__main__":
    main()
