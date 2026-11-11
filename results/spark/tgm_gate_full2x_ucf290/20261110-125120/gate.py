"""TGM-area gate for full-frame 2x on mix + KL (Eric, 2026-11-10): a window takes its 2x score when its TGM hot_area
< 0.045 (the fov2 max_area, fixed before this run), else its 1x score. Writes gated.jsonl and prints zoom rates.
Usage: gate.py <1x scores> <2x scores> <out dir>"""
import json
import sys
from pathlib import Path

MAX_AREA = 0.045
draft_path, zoom_path, out_dir = map(Path, sys.argv[1:4])
draft = [json.loads(line) for line in draft_path.read_text().splitlines()]
zoom = {(r["video"], r["window"]): r["score"] for r in map(json.loads, zoom_path.read_text().splitlines())}


def group(row: dict) -> str:
    if row["label"] == 1:
        return "event"
    return "normal-video negative" if "Normal" in row["video"] else "anomaly-video negative"


counts, zoomed = {}, {}
with (out_dir / "gated.jsonl").open("w") as out:
    for row in draft:
        take_zoom = row["hot_area"] < MAX_AREA
        g = group(row)
        counts[g] = counts.get(g, 0) + 1
        zoomed[g] = zoomed.get(g, 0) + int(take_zoom)
        score = zoom[(row["video"], row["window"])] if take_zoom else row["score"]
        out.write(json.dumps(row | {"score": score}) + "\n")
for g in counts:
    print(f"{g:24s} n={counts[g]:6d}  zoomed {zoomed[g] / counts[g]:.1%}")
print(f"all windows zoomed {sum(zoomed.values()) / sum(counts.values()):.1%}")
