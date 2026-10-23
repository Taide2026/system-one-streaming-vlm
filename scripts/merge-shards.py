#!/usr/bin/env python3
"""Merge the shards of one evaluation into one run folder: scripts/merge-shards.py <out dir> <shard dir> ...

Shards are runs of the same config over disjoint video lists (scripts/eval-shards.sh). The merged folder holds the
first shard's config.yaml, the concatenated manifest.json and scores.jsonl (in shard order), and a run.json with the
window total, the maximum peak GPU memory, the first shard's load time and the shards listed. `s1stream report` then
works on it as on a single run.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out, shards = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(shards[0] / "config.yaml", out / "config.yaml")
    manifest, lines, runs = [], [], []
    for shard in shards:
        manifest.extend(json.loads((shard / "manifest.json").read_text()))
        lines.extend((shard / "scores.jsonl").read_text().splitlines())
        runs.append(json.loads((shard / "run.json").read_text()))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "scores.jsonl").write_text("\n".join(lines) + "\n")
    merged = dict(runs[0])
    merged["windows"] = sum(run["windows"] for run in runs)
    merged["peak_gpu_gb"] = max((run.get("peak_gpu_gb") or 0) for run in runs)
    merged["started"] = min(run["started"] for run in runs)
    merged["finished"] = max(run["finished"] for run in runs)
    merged["shards"] = [str(shard) for shard in shards]
    (out / "run.json").write_text(json.dumps(merged, indent=1))
    print(f"{out}: {len(manifest)} videos, {len(lines)} windows from {len(shards)} shards")


if __name__ == "__main__":
    main()
