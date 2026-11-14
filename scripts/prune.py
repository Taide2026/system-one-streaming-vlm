#!/usr/bin/env python3
"""A FastV-style token-pruning baseline on the 2x input (2026-11-13).

Whole frames enlarged 2x, all 2,400 visual tokens enter the language model. In its first softmax-attention layer
(layer index 3; the layers before it are linear-attention layers and have no attention weights) the read-out position's
attention over the visual tokens ranks them; after that layer only the top `--budget` visual tokens stay, the others
are removed from the sequence for all later layers. This is the published recipe (prune by the attention of an early
layer) moved to the earliest layer this model can supply it from. `--budget 0` keeps everything and must reproduce
the ordinary 2x run.

  python scripts/prune.py --config <eval yaml> --out <run dir> --budget 822 [--rule attention|random]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech import Core, upsample  # noqa: E402

from s1stream.config import load_config  # noqa: E402
from s1stream.input_forms.frames import read_frames  # noqa: E402
from s1stream.registry import BACKBONES, DATASETS  # noqa: E402
from s1stream.windows import plan_windows  # noqa: E402


def take(x, sel, length):
    """Subset the sequence axis (the one of size `length`) of a tensor, or of every tensor in a tuple / list."""
    if x is None:
        return None
    if isinstance(x, (tuple, list)):
        return type(x)(take(v, sel, length) for v in x)
    if torch.is_tensor(x):
        for axis in range(x.dim() - 1, -1, -1):
            if x.shape[axis] == length:
                x = x.index_select(axis, sel)
        return x
    return x


def forward(core: Core, inputs: dict, feats: list, budget: int, rule: str, layer_index: int, rng, given=None) -> tuple[float, int]:
    # rule "given": rank the visual tokens by `given` (one score per visual token, prompt order) instead of attention (2026-11-14)
    model, rt = core.model, core.rt
    ids = inputs["input_ids"]
    embeds = model.get_input_embeddings()(ids)
    flat = torch.cat(feats, dim=0).to(embeds.device, embeds.dtype)
    slots, _ = model.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=flat)
    embeds = embeds.masked_scatter(slots, flat)
    mrope = model.compute_3d_position_ids(
        input_ids=ids, image_grid_thw=inputs["image_grid_thw"], video_grid_thw=None, inputs_embeds=embeds,
        attention_mask=inputs["attention_mask"], past_key_values=None, mm_token_type_ids=inputs["mm_token_type_ids"])
    enc = rt.encoder; lm = enc.language_model; present = inputs["attention_mask"]
    length = ids.shape[1]
    rope = lm.rotary_emb(embeds, mrope)
    keys = present.bool()[:, None, None, :]
    scan = enc.recurrent_mask(config=lm.config, inputs_embeds=embeds, attention_mask=present, past_key_values=None, position_ids=None)
    visual = (ids[0] == core.image_token).nonzero()[:, 0]
    store: list = []
    original = torch.nn.functional.scaled_dot_product_attention

    def spy(q, k, v, *args, **kwargs):
        if q.dim() == 4 and q.shape[-2] == length and k.shape[-2] == length and not store:
            ql, kk = q[:, :, -1:, :].float(), k.float()
            if kk.shape[1] != ql.shape[1]:
                kk = kk.repeat_interleave(ql.shape[1] // kk.shape[1], dim=1)
            store.append((ql @ kk.transpose(-1, -2) / math.sqrt(ql.shape[-1])).softmax(-1)[0, :, 0].mean(0))
        return original(q, k, v, *args, **kwargs)

    hidden, kept = embeds, len(visual)
    for li, (layer, softmax) in enumerate(enc.stack):
        watch = budget and li == layer_index
        if watch:
            torch.nn.functional.scaled_dot_product_attention = spy
        try:
            hidden = layer(hidden, position_embeddings=rope, attention_mask=keys if softmax else scan, position_ids=None,
                           past_key_values=None, use_cache=False)
        finally:
            if watch:
                torch.nn.functional.scaled_dot_product_attention = original
        if watch and budget < len(visual):
            if rule == "given":
                order = given.to(visual.device).float().argsort(descending=True, stable=True)[:budget]
            elif rule == "attention":
                assert store, "no softmax attention seen in the pruning layer"
                order = store[0][visual].argsort(descending=True)[:budget]
            else:
                order = torch.as_tensor(rng.choice(len(visual), budget, replace=False), device=visual.device)
            sel = torch.ones(length, dtype=torch.bool, device=ids.device); sel[visual] = False; sel[visual[order]] = True
            sel = sel.nonzero()[:, 0]
            hidden = hidden.index_select(1, sel); rope = take(rope, sel, length); keys = take(keys, sel, length); scan = take(scan, sel, length)
            kept = budget
    return core.readout(lm.norm(hidden)[:, -1]), kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=822); ap.add_argument("--rule", default="attention", choices=["attention", "random"])
    ap.add_argument("--layer", type=int, default=3); ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.yaml").write_text(yaml.safe_dump({**config.raw, "prune": {"budget": args.budget, "rule": args.rule, "layer": args.layer}}, sort_keys=False))
    backbone = BACKBONES.build(config.backbone); core = Core(backbone, config.question)
    items = list(DATASETS.build(config.dataset).items())
    rng = np.random.default_rng(0); n = 0
    with (args.out / "scores.jsonl").open("w") as stream, torch.inference_mode(), backbone.code.sdpa_backends(core.rt.device):
        for item in items:
            for window in plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames):
                images = read_frames(item.path, window.frame_indices)
                big = core.encode(upsample(images, 2.0, "bicubic")); feats = core.vision(big)
                margin, kept = forward(core, big, feats, args.budget, args.rule, args.layer, rng)
                row = {"video": item.name, "window": window.index, "start_s": window.start_s, "end_s": window.end_s, "label": window.label,
                       "score": 1 / (1 + math.exp(-margin)), "margin": margin, "tokens": int(kept)}
                if n < 3:
                    print(f"window {n}: pruned margin {margin:+.4f}  untouched 2x {core.language(big, feats):+.4f}  kept {kept}", flush=True)
                stream.write(json.dumps(row) + "\n"); n += 1
                if n % 2000 == 0:
                    stream.flush(); print(f"{n} windows", flush=True)
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break
    (args.out / "run.json").write_text(json.dumps({"windows": n, "finished": datetime.now(timezone.utc).isoformat()}))
    print("PRUNE_DONE", n, flush=True)


if __name__ == "__main__":
    main()
