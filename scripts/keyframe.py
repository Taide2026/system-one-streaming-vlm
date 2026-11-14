#!/usr/bin/env python3
"""Keyframe + motion tokens (2026-11-12): encode the scene once, then only what moves.

A surveillance camera does not move, so the eight frames of a window repeat the same background eight times. Layout:
  key frames   every token of the frame (fine 2x tokens, or one coarse token per 2x2 block)
  other frames only the tokens of the blocks on the moving region (touch / loose), optionally plus the coarse tokens
All tokens keep their own (frame, row, column) positions of the 2x layout; the dropped ones are simply absent.
This script measures the EFFECT only: every frame still goes through the vision tower at 2x and tokens are dropped
before the language model. If a layout holds up, the cost version encodes one full frame and small crops.

  python scripts/keyframe.py --config <eval yaml> --out <run dir> --key 3 --key-layout fine --rest touch
      --key          indices of the key frames (0..7), e.g. "3" or "0 7"; "" for none
      --key-layout   fine | coarse
      --rest         touch | loose | none      blocks of the other frames that keep their fine tokens
      --rest-coarse  1: the other frames also keep one coarse token per block (0: nothing outside the fine blocks)
Writes scores.jsonl (rows as the runner's, plus tokens) and run.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech import Core, tgm_mask, upsample  # noqa: E402

from s1stream.config import load_config  # noqa: E402
from s1stream.input_forms.frames import read_frames  # noqa: E402
from s1stream.registry import BACKBONES, DATASETS  # noqa: E402
from s1stream.windows import plan_windows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--key", type=int, nargs="*", default=[3]); ap.add_argument("--key-layout", default="fine", choices=["fine", "coarse"])
    ap.add_argument("--rest", default="touch", choices=["touch", "loose", "none"]); ap.add_argument("--rest-coarse", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.yaml").write_text(yaml.safe_dump({**config.raw, "keyframe": {k: v for k, v in vars(args).items() if k not in ("config", "out")}}, sort_keys=False, default_flow_style=None))
    backbone = BACKBONES.build(config.backbone); core = Core(backbone, config.question)
    items = list(DATASETS.build(config.dataset).items())
    n = 0
    with (args.out / "scores.jsonl").open("w") as stream, torch.inference_mode(), backbone.code.sdpa_backends(core.rt.device):
        for item in items:
            for window in plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames):
                images = read_frames(item.path, window.frame_indices)
                big = core.encode(upsample(images, 2.0, "bicubic")); feats = core.vision(big)
                g2 = big["grids"][0]; hb, wb = math.ceil(g2[0] / 2), math.ceil(g2[1] / 2)
                if args.rest == "none":
                    fine = np.zeros((hb, wb), bool)
                else:
                    fine = cv2.resize(tgm_mask(images), (wb, hb), interpolation=cv2.INTER_AREA) > 0.05
                    if args.rest == "loose":
                        fine = cv2.dilate(fine.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                up = lambda t: t.repeat_interleave(2, 0).repeat_interleave(2, 1)[:g2[0], :g2[1]]
                ft = up(torch.as_tensor(fine, device=feats[0].device))
                anchor = torch.zeros(g2, dtype=torch.bool, device=feats[0].device); anchor[0::2, 0::2] = True
                full = torch.ones(g2, dtype=torch.bool, device=feats[0].device)
                toks, keep = [], []
                for k, f in enumerate(feats):
                    x = f.view(g2[0], g2[1], -1)
                    c = torch.nn.functional.avg_pool2d(x.permute(2, 0, 1)[None].float(), 2, 2, ceil_mode=True, count_include_pad=False)[0]
                    rep = up(c.permute(1, 2, 0)).to(x.dtype)
                    if k in args.key:
                        fine_k, keep_k = (full, full) if args.key_layout == "fine" else (ft, ft | anchor)
                    else:
                        fine_k, keep_k = ft, (ft | anchor) if args.rest_coarse else ft
                    toks.append(torch.where(fine_k[..., None], x, rep).reshape(g2[0] * g2[1], -1)); keep.append(keep_k)
                if not any(bool(k.any()) for k in keep):            # nothing left: fall back to one coarse frame
                    keep[len(keep) // 2] = anchor
                margin = core.language(big, toks, keep=keep)
                tokens = int(sum(int(k.sum()) for k in keep))
                stream.write(json.dumps({"video": item.name, "window": window.index, "start_s": window.start_s, "end_s": window.end_s,
                                         "label": window.label, "score": 1 / (1 + math.exp(-margin)), "margin": margin,
                                         "tokens": tokens, "fine_blocks": int(fine.sum())}) + "\n")
                n += 1
                if n % 2000 == 0:
                    stream.flush(); print(f"{n} windows", flush=True)
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break
    (args.out / "run.json").write_text(json.dumps({"windows": n, "finished": datetime.now(timezone.utc).isoformat()}))
    print("KEY_DONE", n, flush=True)


if __name__ == "__main__":
    main()
