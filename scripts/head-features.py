#!/usr/bin/env python3
"""Per-head features at the answer position (2026-10-28), for a HeadHunt-style probe (deck p. 46):

  uv run --frozen python scripts/head-features.py --backbone d3 --asset d3-nano --prompt v1 \\
      --source <run dir with scores.jsonl + manifest.json | a training run's records.jsonl> --out <features.npz> [--limit N]

One forward pass per window, exactly as the runner (frames -> input form -> backbone.forward), with forward pre-hooks
on every text layer's `self_attn.o_proj` (softmax attention) and `linear_attn.out_proj` (Gated DeltaNet): their input
at the last position is the concatenation of that layer's heads, reshaped to (heads, dim). The final norm's output at
the last position is h_T. Saved (float16): attn (N, L_attn, H, D), linear (N, L_lin, H', D'), hT (N, d), plus video,
window, label (test: the runner's window label; train: y >= 0.5) and y. The model is frozen; nothing is trained here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import s1stream.plugins  # noqa: E402,F401
from s1stream.assets import resolve  # noqa: E402
from s1stream.input_forms.frames import read_frames  # noqa: E402
from s1stream.prompt import PROMPTS  # noqa: E402
from s1stream.registry import BACKBONES  # noqa: E402
from s1stream.types import ModelInput, Question, VideoItem  # noqa: E402
from s1stream.windows import plan_windows  # noqa: E402


def hf_model(backbone) -> torch.nn.Module:
    """The Hugging Face model inside a backbone plugin: d3's runtime.backbone, qwen_first_token's model."""
    model = backbone.model
    return model.backbone if hasattr(model, "backbone") and not isinstance(model, torch.nn.Module) else model


def windows_of(source: Path, root: Path | None, length_s: float, stride_s: float, frames: int, limit: int | None):
    """(video, window index, path, frame_indices, label, y) per window of the source."""
    out = []
    if source.name == "records.jsonl":  # a training run: records carry path and frame indices
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out.append((r["video"], r["window"], Path(r["path"]), tuple(r["frame_indices"]), int(r["y"] >= 0.5), float(r["y"])))
    else:  # an eval run: its manifest + scores
        manifest = {v["name"]: v for v in json.loads((source / "manifest.json").read_text())}
        planned: dict[str, list] = {}
        for line in (source / "scores.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            m = manifest[r["video"]]
            if r["video"] not in planned:
                item = VideoItem(m["name"], (root or Path()) / m["path"], m["category"], m["fps"], m["n_frames"],
                                 tuple(tuple(e) for e in m["events"]))
                planned[r["video"]] = plan_windows(item, length_s, stride_s, frames)
            w = planned[r["video"]][r["window"]]
            out.append((r["video"], r["window"], (root or Path()) / m["path"], w.frame_indices, int(r["label"]), float(r["label"])))
    return out[:limit] if limit else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["d3", "qwen_first_token"])
    ap.add_argument("--asset", required=True)
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None, help="asset root the manifest paths are relative to (eval runs)")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--length-s", type=float, default=2.0)
    ap.add_argument("--stride-s", type=float, default=2.0)
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    p = PROMPTS[args.prompt]
    question = Question(state=p["state"], text=p["text"], options=tuple(p["options"]), positive=p["positive"])
    spec = {"name": args.backbone, "asset": args.asset, "path": str(resolve(args.asset)), "memory_cap_gb": 40}
    if args.adapter:
        spec["adapter"] = args.adapter
    backbone = BACKBONES.build(spec)
    model = hf_model(backbone)

    captured: dict[str, torch.Tensor] = {}
    hooked = []
    for name, module in model.named_modules():
        if name.endswith("self_attn.o_proj") or name.endswith("linear_attn.out_proj"):
            module.register_forward_pre_hook(lambda m, inp, name=name: captured.__setitem__(name, inp[0][:, -1].detach().float().cpu()))
            hooked.append(name)
        elif name.endswith(".norm") and ".layers." not in name and "visual" not in name and "vision" not in name:
            module.register_forward_hook(lambda m, inp, out, name=name: captured.__setitem__("hT", out[:, -1].detach().float().cpu()))
            hooked.append(name)
    attn_names = [n for n in hooked if n.endswith("o_proj")]
    lin_names = [n for n in hooked if n.endswith("out_proj")]
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    H, D = int(cfg.num_attention_heads), int(cfg.head_dim)
    Hl, Dl = int(cfg.linear_num_value_heads), int(cfg.linear_value_head_dim)
    print(f"hooks: {len(attn_names)} attention layers ({H} x {D}), {len(lin_names)} DeltaNet layers ({Hl} x {Dl}), "
          f"final norm {[n for n in hooked if n.endswith('.norm')]}", flush=True)

    rows = windows_of(args.source, args.root, args.length_s, args.stride_s, args.frames, args.limit)
    print(f"{len(rows)} windows from {args.source}", flush=True)
    attn, lin, hT, scores = [], [], [], []
    t0 = time.perf_counter()
    for i, (video, window, path, frame_indices, label, y) in enumerate(rows):
        captured.clear()
        inp = ModelInput(images=read_frames(path, frame_indices), text=None)
        with torch.inference_mode():
            out = backbone.forward(inp, question)
        attn.append(np.stack([captured[n][0].numpy().reshape(H, D) for n in attn_names]).astype(np.float16))
        lin.append(np.stack([captured[n][0].numpy().reshape(Hl, Dl) for n in lin_names]).astype(np.float16))
        hT.append(captured["hT"][0].numpy().astype(np.float16))
        logits = np.asarray(out.logits, dtype=np.float32).flatten() if out.logits is not None else np.zeros(2, np.float32)
        scores.append(logits[:len(question.options)])
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(rows)} {(time.perf_counter() - t0) / (i + 1):.2f} s/window", flush=True)
    np.savez_compressed(args.out, attn=np.stack(attn), linear=np.stack(lin), hT=np.stack(hT), logits=np.stack(scores),
                        video=np.array([r[0] for r in rows]), window=np.array([r[1] for r in rows]),
                        label=np.array([r[4] for r in rows]), y=np.array([r[5] for r in rows]),
                        attn_layers=np.array(attn_names), linear_layers=np.array(lin_names), options=np.array(question.options))
    print(f"saved {args.out}: attn {np.stack(attn).shape}, linear {np.stack(lin).shape}, hT {np.stack(hT).shape}")


if __name__ == "__main__":
    main()
