#!/usr/bin/env python3
"""Hard-negative mining on the TRAIN split (2026-11-02): the normal-video windows the current model scores
highest - "suspicious-looking normal behaviour". Root cause of fov2's thin AP gain: foveation lifts the top
false positives as much as the true positives (+0.096 vs +0.071 mean shift), because training only ever showed
it randomly-sampled (boring) foveated normals. These mined windows go back into the mix as foveated negatives.
TRAIN videos only - the test set is never touched.

  uv run --frozen python scripts/mine-hardneg.py --hivau <ucf_database_train.json> --root <ucf-crime asset> \
      --adapter <floor run> --out <records.jsonl> [--n 2500] [--stride 4.0] [--per-video 40]

Scores every normal train video at `stride`, keeps the top `n` windows overall (capped per video), and writes
them in the trainer's records.jsonl schema with y = 0."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import s1stream.plugins  # noqa: E402,F401
from s1stream.assets import resolve  # noqa: E402
from s1stream.input_forms.frames import read_frames  # noqa: E402
from s1stream.prompt import PROMPTS  # noqa: E402
from s1stream.registry import BACKBONES  # noqa: E402
from s1stream.traindata import Record, read_hivau  # noqa: E402
from s1stream.types import ModelInput, Question  # noqa: E402


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--hivau", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--asset", default="d3-nano")
    ap.add_argument("--backbone", default="d3")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--stride", type=float, default=4.0, help="coarser than 2 s: mining, not evaluation")
    ap.add_argument("--per-video", type=int, default=40)
    args = ap.parse_args()
    p = PROMPTS["v1"]
    question = Question(state=p["state"], text=p["text"], options=tuple(p["options"]), positive=p["positive"])
    spec = {"name": args.backbone, "asset": args.asset, "path": str(resolve(args.asset)), "memory_cap_gb": 40,
            "adapter": args.adapter}
    backbone = BACKBONES.build(spec)
    items = [it for it in read_hivau(args.hivau, args.root) if not it.events]
    print(f"{len(items)} normal train videos", flush=True)
    scored: list[tuple[float, Record]] = []
    for vi, item in enumerate(items):
        per: list[tuple[float, Record]] = []
        t, w = 0.0, 0
        while t + 2.0 <= item.duration_s:
            idx = tuple(min(int(round((t + (k + 0.5) * 0.25) * item.fps)), item.n_frames - 1) for k in range(8))
            try:
                images = read_frames(item.path, idx)
            except Exception:
                break
            with torch.inference_mode():
                out = backbone.forward(ModelInput(images=images, text=None), question)
            z = np.asarray(out.logits, dtype=np.float32).flatten()[:2]
            e = np.exp(z - z.max())
            score = float(e[0] / e.sum())
            per.append((score, Record(video=item.name, path=str(item.path), category="Normal",
                                      window=w, start_s=t, end_s=t + 2.0, frame_indices=idx, y=0.0)))
            t += args.stride
            w += 1
        per.sort(key=lambda x: -x[0])
        scored.extend(per[:args.per_video])
        if (vi + 1) % 25 == 0:
            print(f"{vi + 1}/{len(items)} videos, pool {len(scored)}", flush=True)
    scored.sort(key=lambda x: -x[0])
    keep = scored[:args.n]
    with args.out.open("w") as f:
        for s, r in keep:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"wrote {len(keep)} hard negatives (p from {keep[-1][0]:.3f} to {keep[0][0]:.3f}) -> {args.out}")


if __name__ == "__main__":
    main()
