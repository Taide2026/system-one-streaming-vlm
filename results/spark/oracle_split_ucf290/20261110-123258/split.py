"""Split layouts of mix + KL (floor_d3nano_c) scores (2026-11-09): event windows (label 1) and negative windows
(label 0) each take the 1x (draft) or the 2x (B) score. Writes oracle.jsonl (events 2x, negatives 1x) and
negonly.jsonl (events 1x, negatives 2x) for scripts/layout-eval.py."""
import json
import sys
from pathlib import Path

draft_path, zoom_path, out_dir = map(Path, sys.argv[1:4])
draft = [json.loads(line) for line in draft_path.read_text().splitlines()]
zoom = {(r["video"], r["window"]): r["score"] for r in map(json.loads, zoom_path.read_text().splitlines())}


def write(name: str, event_from_zoom: bool) -> None:
    with (out_dir / name).open("w") as out:
        for row in draft:
            take_zoom = (row["label"] == 1) == event_from_zoom
            score = zoom[(row["video"], row["window"])] if take_zoom else row["score"]
            out.write(json.dumps(row | {"score": score}) + "\n")


write("oracle.jsonl", event_from_zoom=True)
write("negonly.jsonl", event_from_zoom=False)
