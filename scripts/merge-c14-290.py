"""Join a 14-class run on the 117 short UCF-Crime test videos with one on the 173 long ones into a run on all 290.

Usage (repo root): uv run python scripts/merge-c14-290.py <model>   e.g. d3_nano
  short: latest results/spark/<model>_ucf_short_c14_f8/<time>   (117 videos, <= 49.9 s)
  long:  latest results/spark/<model>_ucf_long_c14_f8/<time>    (173 videos, > 49.9 s)
  grid:  latest results/spark/<model>_ucf_test_f8/<time>        (Yes / No on all 290): its (video, window, start_s, end_s)
         must equal the joined rows', so the 290 run sees the same windows as the Yes / No one.
Writes results/spark/<model>_ucf_test_c14_f8/<time of the long run>/ with config.yaml (the long config, renamed, dataset of
all 290), manifest.json and scores.jsonl in the Yes / No run's video order, run.json (merged_from, peak_gpu_gb the
larger one, load_s of the long run); then s1stream.report writes report.json.
"""
import json
import sys
from pathlib import Path

import yaml

from s1stream.report import report


def latest(config: str) -> Path:
    runs = sorted(d for d in Path("results", "spark", config).iterdir() if (d / "run.json").exists())
    if not runs:
        sys.exit(f"no finished run of {config}")
    return runs[-1]


def rows_of(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "scores.jsonl").read_text().splitlines()]


model = sys.argv[1]
short, long, grid = latest(f"{model}_ucf_short_c14_f8"), latest(f"{model}_ucf_long_c14_f8"), latest(f"{model}_ucf_test_f8")
order = [video["name"] for video in json.loads((grid / "manifest.json").read_text())]
manifest = {video["name"]: video for run in (short, long) for video in json.loads((run / "manifest.json").read_text())}
assert sorted(manifest) == sorted(order), "the short and long videos are not the 290 of the Yes / No run"
by_video: dict[str, list[dict]] = {}
for run in (short, long):
    for row in rows_of(run):
        by_video.setdefault(row["video"], []).append(row)
rows = [row for name in order for row in sorted(by_video[name], key=lambda row: row["window"])]
key = lambda row: (row["video"], row["window"], row["start_s"], row["end_s"])
assert [key(row) for row in rows] == [key(row) for row in rows_of(grid)], "window grid differs from the Yes / No run"

out = Path("results", "spark", f"{model}_ucf_test_c14_f8", long.name)
out.mkdir(parents=True, exist_ok=False)
config = yaml.safe_load((long / "config.yaml").read_text())
config["name"] = f"{model}_ucf_test_c14_f8"
config["dataset"].update(n_anomaly=140, n_normal=150, max_duration_s=4000)
config["dataset"].pop("min_duration_s", None)
(out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
(out / "manifest.json").write_text(json.dumps([manifest[name] for name in order], indent=1))
(out / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
runs = [json.loads((run / "run.json").read_text()) for run in (short, long)]
(out / "run.json").write_text(json.dumps({
    "name": config["name"], "merged_from": [str(short), str(long)],
    "peak_gpu_gb": max(r["peak_gpu_gb"] or 0 for r in runs), "load_s": runs[1]["load_s"], "windows": len(rows),
    "runs": runs}, indent=1))
result = report(out)
print(f"{out}: {result['windows']} windows, {result['videos']} videos, frame AUC {result['auc']['frame']:.4f}")
