#!/usr/bin/env python3
"""Extract the 8 frames of every window of the 390 UCF train videos of the V-JEPA 2 head (2026-11-08). CPU only.

Same output layout as scripts/extract-frames.py (<frames-dir>/<video>.npy uint8 [n_windows, 8, H, W, 3] RGB, and
<out>/windows.jsonl), so scripts/sr-score.py can score them. The dataset plugin ucf_crime_train exists only on
experiment/encoder-probing; it is loaded from the path in config.yaml. Frames are read with this branch's
input_forms.frames.read_frames (pixel-equal to the old reader: results/spark/frame_read_check).
At the end, the windows are checked against the V-JEPA 2 training meta: same (video, window, start_s), same count,
and label == holmes.

  uv run --frozen python results/spark/train_zoom_scores_ucf390/<run>/extract_train.py --config <run>/config.yaml \
      --frames-dir $S1_FEATURES/frames/frames_ucf390train_f8/<run> [--limit-videos N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
from s1stream.assets import resolve  # noqa: E402
from s1stream.input_forms.frames import read_frames  # noqa: E402
from s1stream.windows import plan_windows  # noqa: E402

EXTRACT = REPO / "scripts/extract-frames.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_items(config: dict) -> list:
    plugin = load("ucf_crime_train_plugin", Path(config["plugin"]))
    d = config["dataset"]
    # holmes-vau-ucf-train is not in this branch's assets.lock; it is the folder $S1_ASSETS/<name> (encoder-probing).
    return plugin.UcfCrimeTrain(resolve(d["asset"]), d["holmes_asset"], d["max_duration_s"], d["per_class"],
                                d["n_normal"], d["seed"], resolver=holmes_folder).items()


def holmes_folder(name: str) -> Path:
    return Path(os.environ["S1_ASSETS"]) / name


def check(rows: list[dict], meta_paths: list[str]) -> None:
    meta = [json.loads(line) for p in meta_paths for line in Path(p).read_text().splitlines()]
    mine = {(r["video"], r["window"]): r for r in rows}
    theirs = {(m["video"], m["window"]): m for m in meta}
    if mine.keys() != theirs.keys():
        raise SystemExit(f"windows differ: {len(mine.keys() - theirs.keys())} extra, {len(theirs.keys() - mine.keys())} missing")
    bad = [k for k, m in theirs.items() if mine[k]["start_s"] != m["start_s"] or mine[k]["label"] != m["holmes"]]
    if bad:
        raise SystemExit(f"{len(bad)} windows differ in start_s or label, first {bad[0]}")
    print(f"check: {len(rows)} windows equal the V-JEPA 2 training meta (start_s and label)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--limit-videos", type=int, default=None)
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    hot_area = load("extract_frames", EXTRACT).hot_area
    items = train_items(config)
    if args.limit_videos is not None:
        items = items[:args.limit_videos]
    w = config["window"]
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    manifest = [{"name": i.name, "category": i.category, "fps": i.fps, "n_frames": i.n_frames,
                 "events": [list(e) for e in i.events]} for i in items]
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    rows = []
    for n, item in enumerate(items, 1):
        t0 = time.perf_counter()
        stack = []
        for window in plan_windows(item, w["length_s"], w["stride_s"], w["frames"]):
            arrays = [np.asarray(im) for im in read_frames(item.path, window.frame_indices)]
            stack.append(np.stack(arrays))
            rows.append({"video": window.video, "window": window.index, "start_s": window.start_s,
                         "end_s": window.end_s, "label": window.label, "frame_indices": list(window.frame_indices),
                         "hot_area": hot_area(arrays)})
        np.save(args.frames_dir / f"{item.name}.npy", np.stack(stack))
        print(f"[{n}/{len(items)}] {item.name}: {len(stack)} windows, {time.perf_counter() - t0:.1f} s", flush=True)
    (HERE / "windows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if args.limit_videos is None:
        check(rows, config["check_meta"])


if __name__ == "__main__":
    main()
