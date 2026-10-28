#!/usr/bin/env python3
"""UCF101 test clips from the Hugging Face mirror flwrlabs/ucf101 (2026-10-28): crcv.ucf.edu is unreachable from
pro6000. The mirror stores every frame as a JPEG row (image, video_id, clip_id, frame, label; official split 1), so this
script downloads the test shards (30 parquet files, ~15 GB) and writes each clip back as an mp4 at 25 fps:

  uv run --with huggingface_hub --with pyarrow --with opencv-python-headless --with numpy --with pillow \\
      python scripts/ucf101-from-hf.py <root> [--shards N]

<root>/UCF-101/<Class>/<clip_id>.mp4 and <root>/ucfTrainTestlist/testlist01.txt (one "<Class>/<clip_id>.mp4" per clip),
the layout s1stream.datasets.ucf101 reads. The class is taken from clip_id (v_<Class>_gXX_cYY) and matched to
s1stream.prompt.UCF101_CLASSES ignoring case (the mirror writes HandStandPushups). Frames of a clip are not guaranteed to
sit in one shard, so a clip is written when its frames are complete at the end; partial clips across shard boundaries are
buffered. UCF101 is 25 fps, 320 x 240.
"""
from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1stream.prompt import UCF101_CLASSES  # noqa: E402

REPO = "flwrlabs/ucf101"
FPS = 25.0
BY_LOWER = {c.lower(): c for c in UCF101_CLASSES}


def class_of(clip_id: str) -> str:
    raw = clip_id[2:].rsplit("_g", 1)[0] if clip_id.startswith("v_") else clip_id
    if raw.lower() not in BY_LOWER:
        raise SystemExit(f"Unknown class in {clip_id!r}: {raw}")
    return BY_LOWER[raw.lower()]


def write_clip(root: Path, clip_id: str, frames: dict[int, bytes]) -> str:
    cls = class_of(clip_id)
    out = root / "UCF-101" / cls / f"{clip_id}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for index in sorted(frames):
        image = np.array(Image.open(io.BytesIO(frames[index])).convert("RGB"))[:, :, ::-1]
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (image.shape[1], image.shape[0]))
        writer.write(np.ascontiguousarray(image))
    writer.release()
    return f"{cls}/{clip_id}.mp4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--shards", type=int, default=None, help="only the first N test shards (a smoke test)")
    args = ap.parse_args()
    shards = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith("data/test-"))
    if args.shards:
        shards = shards[:args.shards]
    pending: dict[str, dict[int, bytes]] = defaultdict(dict)
    written: list[str] = []
    for shard in shards:
        path = hf_hub_download(REPO, shard, repo_type="dataset")
        table = pq.read_table(path, columns=["image", "clip_id", "frame"])
        clips, frames, images = table.column("clip_id").to_pylist(), table.column("frame").to_pylist(), table.column("image").to_pylist()
        seen_here: set[str] = set()
        for clip, frame, image in zip(clips, frames, images):
            pending[clip][int(frame)] = image["bytes"]
            seen_here.add(clip)
        # clips that did not continue into this shard are complete
        for clip in [c for c in pending if c not in seen_here]:
            written.append(write_clip(args.root, clip, pending.pop(clip)))
        print(f"{shard}: {len(seen_here)} clips in shard, {len(written)} written, {len(pending)} open", flush=True)
    for clip in list(pending):
        written.append(write_clip(args.root, clip, pending.pop(clip)))
    (args.root / "ucfTrainTestlist").mkdir(parents=True, exist_ok=True)
    (args.root / "ucfTrainTestlist" / "testlist01.txt").write_text("\n".join(sorted(written)) + "\n")
    print(f"{len(written)} clips written to {args.root}")


if __name__ == "__main__":
    main()
