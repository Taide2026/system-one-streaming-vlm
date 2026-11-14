#!/usr/bin/env python3
"""Fuse two runs' per-window scores (2026-11-05): mean / max / geometric mean, written as derived run dirs so
scripts/jev-hard.py and tier-bootstrap.py can score them.  python fuse-scores.py <runA> <runB> <out prefix>"""
import json, sys, shutil, math
from pathlib import Path
a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ra = {(r["video"], r["window"]): r for r in map(json.loads, a.joinpath("scores.jsonl").read_text().splitlines()) if r}
rb = {(r["video"], r["window"]): r for r in map(json.loads, b.joinpath("scores.jsonl").read_text().splitlines()) if r}
keys = sorted(set(ra) & set(rb))
for name, f in (("mean", lambda x, y: (x + y) / 2), ("max", max), ("gmean", lambda x, y: math.sqrt(x * y))):
    d = Path(f"{out}_{name}"); d.mkdir(parents=True, exist_ok=True)
    shutil.copy(a / "manifest.json", d / "manifest.json")
    with open(d / "scores.jsonl", "w") as fh:
        for k in keys:
            r = dict(ra[k]); r["score"] = f(ra[k]["score"], rb[k]["score"]); fh.write(json.dumps(r) + "\n")
    print(name, d, len(keys))
