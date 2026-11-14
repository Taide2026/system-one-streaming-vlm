#!/usr/bin/env python3
"""ComplexVAD -> the frames our windows read, at several real resolutions (2026-11-13). CPU only.

A test video is 1920x1080. For every 2 s window the 8 frames the windower samples (time (k + 0.5) / 4 s) are decoded
once from the native file and written, area-downscaled, to one Motion-JPEG file per resolution (quality 95, every
frame a key frame, so seeking is exact). The stored file runs at 8 fps with each sampled frame written twice: stored
frame j is native time j / 8 s, and the windower's indices (16 w + 2 k + 1) land on the sampled frames exactly.

  python3 cvad-prep.py <root> [--jobs 6]      reads <root>/Test/*.mp4, writes <root>/v{180,360,720}/<id>.avi, <root>/meta/<id>.json
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

import os

ALL = {"v180": (320, 180), "v360": (640, 360), "v720": (1280, 720)}
SIZES = {k: v for k, v in ALL.items() if k in os.environ.get("CVAD_SIZES", "v180,v360,v720").split(",")}


def one(path: str, root: str) -> str:
    root_p, name = Path(root), Path(path).stem
    meta = root_p / "meta" / f"{name}.json"
    if meta.is_file():
        return f"have {name}"
    cap = cv2.VideoCapture(path)
    fps, n = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    windows = int((n / fps + 1e-9) // 2)
    targets = [round((2 * w + (k + 0.5) / 4) * fps) for w in range(windows) for k in range(8)]
    writers = {}
    for tag, size in SIZES.items():
        (root_p / tag).mkdir(parents=True, exist_ok=True)
        wr = cv2.VideoWriter(str(root_p / tag / f"{name}.tmp.avi"), cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc(*"MJPG"), 8.0, size)
        wr.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
        writers[tag] = wr
    pos, done = 0, 0
    for t in targets:
        while pos < t:
            if not cap.grab():
                break
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if not ok:
            break
        for tag, size in SIZES.items():
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            writers[tag].write(small); writers[tag].write(small)
        done += 1
    cap.release()
    for tag, wr in writers.items():
        wr.release()
    done -= done % 8                                                  # whole windows only
    if done == 0:
        return f"FAILED {name}"
    for tag in SIZES:
        (root_p / tag / f"{name}.tmp.avi").rename(root_p / tag / f"{name}.avi")
    meta.parent.mkdir(exist_ok=True)
    meta.write_text(json.dumps({"fps": fps, "n_frames": n, "native": [int(cap_w) for cap_w in (1920, 1080)], "stored_frames": 2 * done, "windows": done // 8}))
    return f"done {name} fps {fps:.2f} frames {n} windows {done // 8}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    videos = sorted(str(p) for p in (Path(a.root) / "Test").glob("*.mp4"))
    print(len(videos), "complete videos", flush=True)
    with ProcessPoolExecutor(a.jobs) as pool:
        for msg in pool.map(one, videos, [a.root] * len(videos)):
            print(msg, flush=True)
    print("PREP_DONE", flush=True)
    sys.exit(0)
