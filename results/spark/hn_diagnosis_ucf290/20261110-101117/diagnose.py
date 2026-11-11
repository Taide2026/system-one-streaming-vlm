"""Two checks of floor2x_hn_d3nano_c against mix + KL (floor_d3nano_c), UCF-Crime test 290 (Eric, 2026-11-10).

1. Macro AUC: per anomaly video, the window AUC of event windows against that video's other windows; mean over
   videos. Paired bootstrap over anomaly videos (1,000 draws, seed 20261022).
2. Score by distance: in anomaly videos, the mean window score by distance (in windows) to the nearest event
   window; distance 0 = event window.
Usage: diagnose.py <name>=<scores.jsonl> ... (the first one is the baseline)
"""
import json
import sys
from pathlib import Path

import numpy as np

from s1stream import metrics

DRAWS, SEED = 1000, 20261022
BINS = [(0, 0, "事件 window"), (1, 1, "距離 1"), (2, 3, "距離 2–3"), (4, 10, "距離 4–10"), (11, 10**9, "距離 11 以上")]


def by_video(path: Path) -> dict[str, list[tuple[int, int, float]]]:
    """{video: [(window index, label, score), ...]} sorted by window."""
    out: dict[str, list] = {}
    for row in map(json.loads, path.read_text().splitlines()):
        out.setdefault(row["video"], []).append((row["window"], row["label"], row["score"]))
    return {v: sorted(rows) for v, rows in out.items()}


def video_aucs(videos: dict) -> dict[str, float]:
    """Window AUC per video that has both event and non-event windows."""
    out = {}
    for video, rows in videos.items():
        labels = [label for _, label, _ in rows]
        if 0 < sum(labels) < len(labels):
            out[video] = metrics.auc([score for _, _, score in rows], labels)
    return out


def distance_means(videos: dict) -> dict[str, tuple[int, float]]:
    """{bin name: (windows, mean score)} over anomaly videos."""
    sums = {name: [0, 0.0] for _, _, name in BINS}
    for rows in videos.values():
        events = np.array([w for w, label, _ in rows if label])
        if events.size == 0:
            continue
        for window, _, score in rows:
            distance = int(np.abs(events - window).min())
            for low, high, name in BINS:
                if low <= distance <= high:
                    sums[name][0] += 1
                    sums[name][1] += score
    return {name: (n, total / n) for name, (n, total) in sums.items()}


def main() -> None:
    runs = dict(arg.split("=", 1) for arg in sys.argv[1:])
    names = list(runs)
    videos = {name: by_video(Path(path)) for name, path in runs.items()}
    aucs = {name: video_aucs(videos[name]) for name in names}
    keys = sorted(aucs[names[0]])
    table = {name: np.array([aucs[name][k] for k in keys]) for name in names}
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, len(keys), len(keys)) for _ in range(DRAWS)]
    print(f"Macro AUC over {len(keys)} anomaly videos (window level)")
    for name in names:
        diff = [table[name][d].mean() - table[names[0]][d].mean() for d in draws]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        print(f"  {name:8s} {100 * table[name].mean():.1f}  d[{100 * lo:+.1f}, {100 * hi:+.1f}]")
    print("Mean score by distance to the nearest event window (anomaly videos)")
    means = {name: distance_means(videos[name]) for name in names}
    for _, _, label in BINS:
        n = means[names[0]][label][0]
        cells = "  ".join(f"{name} {means[name][label][1]:.3f}" for name in names)
        print(f"  {label:10s} n={n:5d}  {cells}")


if __name__ == "__main__":
    main()
