"""Frame-level AUC of existing window scores on UCF-Crime test 290, with paired video bootstrap (2026-11-09).

Each frame takes the max score of the windows that cover it (s1stream.metrics.frame_scores); frames no window
covers are left out. Bootstrap: 1,000 draws, seed 20261022, anomaly and normal videos drawn separately, the same
draws for every run, so each interval is the paired difference against the baseline.
Usage: frame-auc.py <manifest.json> <out.json> <baseline name>=<scores.jsonl> <name>=<scores.jsonl> ...
"""
import json
import sys
from pathlib import Path

import numpy as np

from s1stream import metrics
from s1stream.windows import window_frame_range

DRAWS, SEED = 1000, 20261022


def fast_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Same value as metrics.auc (average rank for ties), vectorised for the bootstrap."""
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2
    positive = labels == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    rank_sum = float(average_rank[inverse][positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def video_frames(manifest: list[dict], score_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{video: (frame scores, frame labels)} for one score file."""
    windows: dict[str, list] = {}
    fps = {m["name"]: float(m["fps"]) for m in manifest}
    for row in map(json.loads, score_path.read_text().splitlines()):
        first, last = window_frame_range(row["start_s"], row["end_s"], fps[row["video"]])
        windows.setdefault(row["video"], []).append((first, last, row["score"]))
    out = {}
    for m in manifest:
        video = {"n_frames": int(m["n_frames"]), "events": m["events"], "windows": windows[m["name"]]}
        out[m["name"]] = metrics.frame_scores([video])
    return out


def draw_videos(manifest: list[dict], rng: np.random.Generator) -> list[str]:
    anomaly = [m["name"] for m in manifest if m["events"]]
    normal = [m["name"] for m in manifest if not m["events"]]
    return list(rng.choice(anomaly, len(anomaly))) + list(rng.choice(normal, len(normal)))


def concat(frames: dict, videos: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([frames[v][0] for v in videos]), np.concatenate([frames[v][1] for v in videos])


METRICS = {"auc": fast_auc, "ap": metrics.ap}  # added 2026-11-09: frame AP next to frame AUC


def interval(values) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2])
    runs = dict(arg.split("=", 1) for arg in sys.argv[3:])
    names = list(runs)
    frames = {name: video_frames(manifest, Path(path)) for name, path in runs.items()}
    all_videos = [m["name"] for m in manifest]
    point = {name: {"auc": metrics.auc(*concat(frames[name], all_videos)),
                    "ap": metrics.ap(*concat(frames[name], all_videos))} for name in names}
    rng = np.random.default_rng(SEED)
    boot = {name: {key: [] for key in METRICS} for name in names}
    for _ in range(DRAWS):
        videos = draw_videos(manifest, rng)
        for name in names:
            scores, labels = concat(frames[name], videos)
            for key, metric in METRICS.items():
                boot[name][key].append(metric(scores, labels))
    result = {}
    for name in names:
        result[name] = {"frames": int(concat(frames[name], all_videos)[0].size)}
        for key in METRICS:
            values = np.array(boot[name][key])
            diff = values - np.array(boot[names[0]][key])
            result[name][key] = {"value": point[name][key], "ci": interval(values), "delta_ci": interval(diff)}
            print(f"{name:10s} frame {key.upper():3s} {100 * point[name][key]:.2f}  "
                  f"d[{100 * result[name][key]['delta_ci'][0]:+.2f}, {100 * result[name][key]['delta_ci'][1]:+.2f}]",
                  flush=True)
    out_path.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
