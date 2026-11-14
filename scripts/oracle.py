#!/usr/bin/env python3
"""Oracle placement of the fine tokens (2026-11-12): how much could a better choice of blocks give?

The mixed-resolution layout of mech.py (one coarse token per 2x2 block, four fine tokens in chosen blocks, the other
tokens dropped), with the blocks chosen by exhaustive measurement instead of by the motion map. A first-order
(gradient) estimate of a block's effect was tried first and does not track the measured effect (rank correlation
about 0), so every block is measured. For each window of a stratified sample:
  1. 2x frames through the vision tower once; margin with all blocks coarse.
  2. for each of the 80 blocks: margin with only that block fine  ->  effect[b] = margin_b - margin_coarse.
  3. top-k blocks by one of two rules
        abs    largest |effect|: where fine detail moves the decision most (uses no label)
        label  largest effect for a positive window, most negative for a negative one: the placement that helps the
               correct answer most (uses the label: an upper bound for any selector, not a method)
  4. margins of the oracle layouts (rule x k) and, on the same window, of the motion-map layouts (tight, touch,
     loose), random blocks, all coarse and the untouched 2x input.

  python scripts/oracle.py --config <eval yaml> --events <jev-hard-events.jsonl> --a <a finished 1x run of the same
      model> --out <dir> --ks 5 24 --per-group 600 [--shard 0 --nshards 4]
Writes <out>/rows.shard<i>.jsonl: one line per window with its group, label, every layout's margin and token count,
the 80 block effects and the motion-map block masks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech import Core, tgm_mask, upsample  # noqa: E402

from s1stream.config import load_config  # noqa: E402
from s1stream.input_forms.frames import read_frames  # noqa: E402
from s1stream.registry import BACKBONES, DATASETS  # noqa: E402
from s1stream.windows import plan_windows  # noqa: E402


class GradCore(Core):
    def margin_tensor(self, inputs: dict, features: list, keep: list | None = None):
        """Core.language with the graph kept: logit(Yes) - logit(No) as a tensor."""
        model, rt = self.model, self.rt
        ids = inputs["input_ids"]
        embeds = model.get_input_embeddings()(ids)
        flat = torch.cat(features, dim=0).to(embeds.device, embeds.dtype)
        slots, _ = model.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=flat)
        embeds = embeds.masked_scatter(slots, flat)
        mrope = model.compute_3d_position_ids(
            input_ids=ids, image_grid_thw=inputs["image_grid_thw"], video_grid_thw=None, inputs_embeds=embeds,
            attention_mask=inputs["attention_mask"], past_key_values=None, mm_token_type_ids=inputs["mm_token_type_ids"])
        enc = rt.encoder; lm = enc.language_model; present = inputs["attention_mask"]
        if keep is not None:
            sel = torch.ones(ids.shape[1], dtype=torch.bool, device=ids.device)
            sel[(ids[0] == self.image_token).nonzero()[:, 0]] = torch.cat([k.reshape(-1) for k in keep]).to(ids.device)
            embeds, mrope, present = embeds[:, sel], mrope[:, :, sel], present[:, sel]
        rope = lm.rotary_emb(embeds, mrope)
        keys = present.bool()[:, None, None, :]
        scan = enc.recurrent_mask(config=lm.config, inputs_embeds=embeds, attention_mask=present, past_key_values=None, position_ids=None)
        hidden = embeds
        for layer, softmax in enc.stack:
            hidden = layer(hidden, position_embeddings=rope, attention_mask=keys if softmax else scan, position_ids=None,
                           past_key_values=None, use_cache=False)
        last = lm.norm(hidden)[:, -1]
        logits = last.float() @ rt.readout.float().T
        logits = logits[0, :2] / rt.temperature
        return logits[0] - logits[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True); ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 24]); ap.add_argument("--per-group", type=int, default=600)
    ap.add_argument("--cut", type=float, default=0.04)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    key = lambda n: n[:-4] if n.endswith(".mp4") else n
    events = [json.loads(l) for l in args.events.open()]
    by_video = defaultdict(list)
    for e in events:
        by_video[key(e["video"])].append((e["event"][0], e["event"][1], e["area_frac"]))
    A = {(key(r["video"]), r["window"]): r for r in map(json.loads, (args.a / "scores.jsonl").open())}
    positives = defaultdict(list)
    for (v, w), r in A.items():
        if r["label"] == 1:
            positives[v].append(w)
    groups = defaultdict(list)
    for (v, w), r in A.items():
        if r["label"] == 1:
            lo, hi = r["start_s"] * 30, r["end_s"] * 30 - 1
            area = next((a for s, e, a in by_video.get(v, []) if lo <= e and s <= hi), None)
            if area is not None:
                groups["small+" if area < args.cut else "large+"].append((v, w))
        elif v not in positives:
            groups["normal-"].append((v, w))
        else:
            groups["near-" if min(abs(w - p) for p in positives[v]) <= 5 else "far-"].append((v, w))
    rng = np.random.default_rng(2)
    sample = []
    for g in ("small+", "large+", "near-", "far-", "normal-"):
        rows = sorted(groups[g]); idx = rng.choice(len(rows), min(args.per_group, len(rows)), replace=False)
        sample += [(g, *rows[i]) for i in sorted(idx)]
    sample = sample[args.shard::args.nshards]
    if args.limit:
        sample = sample[:: max(len(sample) // args.limit, 1)][:args.limit]

    config = load_config(args.config)
    backbone = BACKBONES.build(config.backbone); core = GradCore(backbone, config.question)
    items = {key(item.name): item for item in DATASETS.build(config.dataset).items()}
    args.out.mkdir(parents=True, exist_ok=True)
    windows = {}
    with (args.out / f"rows.shard{args.shard}.jsonl").open("w") as stream, torch.no_grad(), backbone.code.sdpa_backends(core.rt.device):
        for done, (g, v, w) in enumerate(sample):
            item = items[v]
            if v not in windows:
                windows[v] = {x.index: x for x in plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames)}
            window = windows[v][w]
            images = read_frames(item.path, window.frame_indices)
            big = core.encode(upsample(images, 2.0, "bicubic")); feats = core.vision(big)
            g2 = big["grids"][0]; hb, wb = math.ceil(g2[0] / 2), math.ceil(g2[1] / 2)
            up = lambda t: t.repeat_interleave(2, 0).repeat_interleave(2, 1)[:g2[0], :g2[1]]
            fine = [f.view(g2[0], g2[1], -1) for f in feats]
            rep = [up(torch.nn.functional.avg_pool2d(x.permute(2, 0, 1)[None].float(), 2, 2, ceil_mode=True,
                                                       count_include_pad=False)[0].permute(1, 2, 0)).to(x.dtype) for x in fine]
            anchor = torch.zeros(g2, dtype=torch.bool, device=feats[0].device); anchor[0::2, 0::2] = True

            def run(mask):
                ft = up(torch.as_tensor(mask, device=feats[0].device))
                toks = [torch.where(ft[..., None], x, r).reshape(g2[0] * g2[1], -1) for x, r in zip(fine, rep)]
                return float(core.margin_tensor(big, toks, keep=[ft | anchor] * len(toks))), int(len(toks) * int((ft | anchor).sum()))

            none = np.zeros((hb, wb), bool)
            coarse, tok0 = run(none)
            effect = np.zeros(hb * wb)
            for b in range(hb * wb):
                m = np.zeros(hb * wb, bool); m[b] = True
                effect[b] = run(m.reshape(hb, wb))[0] - coarse
            cov = cv2.resize(tgm_mask(images), (wb, hb), interpolation=cv2.INTER_AREA)
            touch = cov > 0.05
            layouts = {"tight": cov > 0.25, "touch": touch,
                       "loose": cv2.dilate(touch.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)}
            r = np.random.default_rng(abs(hash((v, w))) % (2 ** 32))
            for k in args.ks:
                m = np.zeros(hb * wb, bool); m[r.choice(hb * wb, k, replace=False)] = True; layouts[f"rand_k{k}"] = m.reshape(hb, wb)
                m = np.zeros(hb * wb, bool); m[np.argsort(-np.abs(effect))[:k]] = True; layouts[f"abs_k{k}"] = m.reshape(hb, wb)
                m = np.zeros(hb * wb, bool); m[np.argsort(-effect if window.label == 1 else effect)[:k]] = True; layouts[f"label_k{k}"] = m.reshape(hb, wb)
            row = {"group": g, "video": v, "window": w, "label": window.label, "blocks": [hb, wb],
                   "margin": {"coarse": coarse, "full2x": float(core.margin_tensor(big, feats))},
                   "tokens": {"coarse": tok0, "full2x": int(len(feats) * g2[0] * g2[1])},
                   "effect": [round(float(e), 4) for e in effect],
                   "mask": {n: np.flatnonzero(layouts[n].reshape(-1)).tolist() for n in ("tight", "touch", "loose")}}
            for n, m in layouts.items():
                row["margin"][n], row["tokens"][n] = run(m)
            stream.write(json.dumps(row) + "\n")
            if done % 50 == 0:
                stream.flush(); print(f"shard {args.shard}: {done}/{len(sample)}", flush=True)
    print("ORACLE_DONE", args.shard, len(sample), flush=True)


if __name__ == "__main__":
    main()
