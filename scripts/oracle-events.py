#!/usr/bin/env python3
"""Oracle timing over every annotated event (2026-10-30): if the model saw the event perfectly (8 frames all inside,
full frame), what share of the 156 UCF test events crosses 0.5?

  uv run --frozen python scripts/oracle-events.py --manifest <manifest.json> --root <asset> [--adapter <run>] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import s1stream.plugins  # noqa: E402,F401
from s1stream.assets import resolve  # noqa: E402
from s1stream.prompt import PROMPTS  # noqa: E402
from s1stream.registry import BACKBONES  # noqa: E402
from s1stream.types import ModelInput, Question  # noqa: E402


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--adapter", default=None); ap.add_argument("--asset", default="d3-nano")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    man = json.loads(args.manifest.read_text())
    p = PROMPTS["v1"]
    question = Question(state=p["state"], text=p["text"], options=tuple(p["options"]), positive=p["positive"])
    spec = {"name": "d3", "asset": args.asset, "path": str(resolve(args.asset)), "memory_cap_gb": 40}
    if args.adapter:
        spec["adapter"] = args.adapter
    backbone = BACKBONES.build(spec)
    out = []
    for m in man:
        if not m["events"]:
            continue
        cap = cv2.VideoCapture(str(args.root / m["path"]))
        for e0, e1 in m["events"]:
            t0, t1 = e0 / m["fps"], (e1 + 1) / m["fps"]
            images = []
            for k in range(8):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((t0 + (k + 0.5) * (t1 - t0) / 8) * m["fps"])))
                ok, fr = cap.read()
                if ok:
                    images.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            if len(images) < 4:
                continue
            with torch.inference_mode():
                o = backbone.forward(ModelInput(images=images, text=None), question)
            z = np.asarray(o.logits, dtype=np.float32).flatten()[:2]; e = np.exp(z - z.max())
            out.append({"video": m["name"], "event": [e0, e1], "p": float(e[0] / e.sum())})
        cap.release()
    args.out.write_text(json.dumps(out, indent=1))
    ps = np.array([r["p"] for r in out])
    print(f"{len(out)} events, caught at 0.5: {(ps >= 0.5).sum()} ({100 * (ps >= 0.5).mean():.1f} %), mean p {ps.mean():.2f}")


if __name__ == "__main__":
    main()
