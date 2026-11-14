#!/usr/bin/env python3
"""Build a 'ucf-crime-train' asset on nano4 (2026-11-07): the UCF-Crime TRAINING videos exposed through the same
dataset code as the test set, so that a model can be scored on them (roadmap D1 prerequisite: policies and RL learn
on training-split scores only). Videos are symlinks into the ucf-crime asset; the annotation file is generated from
HIVAU-70k's training events (seconds -> frames; normal videos get no events, whatever HIVAU lists for them).
  python gen-train-asset.py <s1-assets root> <hivau ucf_database_train.json> <repo root>
Also appends a lock entry 'ucf-crime-train' to <repo>/assets.yaml (resolve() needs the name in the lock) and writes
<repo>/ucf-train-manifest.json (names only, for scripts/eval-shards.sh)."""
import json, os, sys
from pathlib import Path

assets, hivau, repo = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
src = assets / "ucf-crime"; dst = assets / "ucf-crime-train"
(dst / "videos").mkdir(parents=True, exist_ok=True); (dst / "annotations").mkdir(exist_ok=True)
h = json.load(open(hivau))
rows, names, missing = [], [], 0
for name, v in sorted(h.items()):
    fname = name + ".mp4"
    normal = name.startswith("Normal")
    cat = "Normal" if normal else v["label"][0]
    folder = "Training_Normal_Videos_Anomaly" if normal else cat
    if not (src / "videos" / folder / fname).exists():
        missing += 1; continue
    fps = float(v["fps"])
    if normal:
        pairs = "-1 -1"
    else:
        ev = [(int(round(a * fps)), int(round(b * fps))) for a, b in v["events"]]
        pairs = "  ".join(f"{a} {b}" for a, b in ev) if ev else "-1 -1"
    rows.append(f"{fname}  {cat}  {pairs}")
    names.append(fname)
# videos: class folders -> the ucf-crime folders; Normal -> the TRAINING normal folder under the test-normal name
for folder in os.listdir(src / "videos"):
    if folder.startswith("Testing_Normal") or folder.startswith("Training_Normal"):
        continue
    link = dst / "videos" / folder
    if not link.exists():
        link.symlink_to(src / "videos" / folder)
nl = dst / "videos" / "Testing_Normal_Videos_Anomaly"
if not nl.exists():
    nl.symlink_to(src / "videos" / "Training_Normal_Videos_Anomaly")
(dst / "annotations" / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").write_text("\n".join(rows) + "\n")
for extra in ("splits", "ReadMe-Anomaly-Detection.txt"):
    p = dst / extra
    if (src / extra).exists() and not p.exists():
        p.symlink_to(src / extra)
lock = repo / "assets.yaml"
if "ucf-crime-train:" not in lock.read_text():
    with lock.open("a") as f:
        f.write("\nucf-crime-train:\n  kind: archive\n  source: local (gen-train-asset.py; UCF-Crime training split via HIVAU-70k)\n  size: 0\n  sha256: {}\n")
(repo / "ucf-train-manifest.json").write_text(json.dumps([{"name": n} for n in names]))
print(f"{len(rows)} training videos ({sum(1 for r in rows if '  Normal  ' in r)} normal), {missing} missing; manifest {repo / 'ucf-train-manifest.json'}")
