#!/usr/bin/env python3
"""UCFCrime2Local boxes -> events with a ground-truth box area (2026-11-05), to check the TGM size proxy of
scripts/tgm-size.py against hand-drawn boxes. No model - CPU.

  uv run --frozen python scripts/c2l-events.py --ann "<UCFCrime2Local>/Txt annotations" --root <ucf-crime asset> --out <dir>

Vatic format, one line per frame: track xmin ymin xmax ymax frame lost occluded generated label. Each video has one
track; a frame with lost == 0 shows the anomaly. An event = a run of visible frames; runs closer than MERGE_GAP
frames are merged, runs shorter than MIN_LEN frames are dropped. box_area = median over the event's visible frames
of (clipped box area) / (frame area). Note: all 100 videos are UCF-Crime train videos, not the 290 test videos.
Outputs:
  <out>/manifest.json   tgm-size.py input: [{name, path, fps, events}]
  <out>/c2l-events.jsonl one line per event: video, event, box_area, n_frames
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MERGE_GAP = 30  # frames (1 s at 30 fps)
MIN_LEN = 16  # frames; shorter runs give no full 2 s probe window anyway


def read_boxes(path: Path) -> dict[int, tuple[int, int, int, int]]:
    """Visible boxes by frame: {frame: (x0, y0, x1, y1)}."""
    boxes = {}
    for line in path.read_text().splitlines():
        f = line.split()
        if f[6] == "0":
            boxes[int(f[5])] = (int(f[1]), int(f[2]), int(f[3]), int(f[4]))
    return boxes


def runs(frames: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of frame numbers, merged across gaps <= MERGE_GAP, kept when >= MIN_LEN frames long."""
    if not frames:
        return []
    out = [[frames[0], frames[0]]]
    for f in frames[1:]:
        if f - out[-1][1] <= MERGE_GAP:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [(a, b) for a, b in out if b - a + 1 >= MIN_LEN]


def box_area(boxes: dict, start: int, end: int, width: int, height: int) -> float:
    """Median clipped box area over the visible frames of [start, end], as a share of the frame."""
    areas = []
    for f in range(start, end + 1):
        if f not in boxes:
            continue
        x0, y0, x1, y1 = boxes[f]
        w = max(0, min(x1, width) - max(x0, 0))
        h = max(0, min(y1, height) - max(y0, 0))
        areas.append(w * h / (width * height))
    return float(np.median(areas))


def video_info(path: Path) -> tuple[float, int, int, int]:
    """fps, n_frames, width, height."""
    cap = cv2.VideoCapture(str(path))
    try:
        return cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), \
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest, rows = [], []
    for ann in sorted(args.ann.glob("*.txt")):
        clip = ann.stem  # e.g. Arrest002
        category = clip.rstrip("0123456789")
        rel = f"videos/{category}/{clip}_x264.mp4"
        fps, n_frames, width, height = video_info(args.root / rel)
        boxes = read_boxes(ann)
        events = [(a, min(b, n_frames - 1)) for a, b in runs(sorted(boxes))]
        manifest.append({"name": f"{clip}_x264.mp4", "path": rel, "fps": fps, "events": [list(e) for e in events]})
        for a, b in events:
            rows.append({"video": f"{clip}_x264.mp4", "event": [a, b],
                         "box_area": box_area(boxes, a, b, width, height), "n_frames": b - a + 1})

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (args.out / "c2l-events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(manifest)} videos, {len(rows)} events")


if __name__ == "__main__":
    main()
