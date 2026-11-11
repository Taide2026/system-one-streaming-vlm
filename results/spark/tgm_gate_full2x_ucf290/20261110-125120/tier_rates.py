"""Zoom rate (hot_area < 0.045) of the event windows per tier, with the tier rule of scripts/layout-eval.py."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
le = __import__("layout-eval")

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
manifest = {m["name"]: m for m in json.loads(Path(sys.argv[2]).read_text())}
events = [json.loads(line) for line in Path(sys.argv[3]).read_text().splitlines()]
zoom = np.array([r["hot_area"] < 0.045 for r in rows])
for tier, (include, y) in le.tier_masks(rows, manifest, events).items():
    pos = include & (y == 1)
    print(f"{tier:6s} events {int(pos.sum()):5d}  zoomed {zoom[pos].mean():.1%}")
