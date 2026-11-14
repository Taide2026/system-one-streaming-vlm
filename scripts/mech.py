#!/usr/bin/env python3
"""Two mechanism experiments on the fine-tuned d3-nano (2026-11-12). The runtime's plain forward path, taken apart
into: processor -> vision tower -> (intervention on the visual tokens) -> language model -> read-out.

pool    A token-pooling intervention as a full evaluation (writes scores.jsonl like the runner).
          --mode identity   2x frames, tokens untouched (must reproduce the ordinary 2x run)
          --mode pool1x     2x frames through the vision tower, each frame's tokens average-pooled to the 1x grid,
                            placed in the 1x prompt: high-resolution features, 1x token count
          --mode poolrep    the same pooled tokens replicated back to the 2x grid, in the 2x prompt:
                            2x token count, no content finer than the 1x grid
layers  Layer-wise comparison of three inputs (1x, bicubic 2x, nearest 2x) on the probe set: per vision block the
        similarity of the token sets, per language-model layer the read-out position's state and the margin the
        read-out gives at that depth (logit lens).

  python scripts/mech.py pool   --config <eval yaml> --mode pool1x --kernel bicubic --out <run dir>
  python scripts/mech.py layers --config <eval yaml> --events <jev-hard-events.jsonl> --a <1x run> --b <2x run> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

import s1stream.plugins  # noqa: F401
from s1stream.backbones.d3 import build_questions
from s1stream.config import load_config
from s1stream.input_forms.frames import read_frames
from s1stream.input_forms.upsample import upsample
from s1stream.registry import BACKBONES, DATASETS
from s1stream.types import ModelInput
from s1stream.windows import plan_windows


class Core:
    def __init__(self, backbone, question) -> None:
        self.wrapper, self.rt, self.question = backbone, backbone.model, question
        self.model = self.rt.backbone
        self.image_token = int(self.model.config.image_token_id)
        assert self.rt.encoder is not None
        self.block_out: list = []
        self.hooks = [b.register_forward_hook(self._hook) for b in self.model.visual.blocks]
        self.capture_blocks = False

    def _hook(self, module, args, output) -> None:
        if self.capture_blocks:
            self.block_out.append((output[0] if isinstance(output, (tuple, list)) else output).detach())

    def encode(self, images: list) -> dict:
        rt = self.rt
        prepared = rt.prepare(self.question.state, build_questions(self.question), list(images), None)
        key = prepared.runnable[0]
        encoded = rt.processor(text=[prepared.texts[key]], images=list(prepared.images), padding=True, return_tensors="pt")
        inputs = {name: value.to(rt.device) for name, value in encoded.items()}
        inputs["grids"] = [(int(h) // 2, int(w) // 2) for _, h, w in inputs["image_grid_thw"].tolist()]
        return inputs

    def vision(self, inputs: dict, blocks: bool = False) -> list:
        self.block_out, self.capture_blocks = [], blocks
        try:
            pooled = self.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True).pooler_output
        finally:
            self.capture_blocks = False
        return list(pooled)

    def language(self, inputs: dict, features: list, per_layer: bool = False, keep: list | None = None):
        """Visual tokens `features` (one tensor per frame) placed in the prompt of `inputs`; returns the margin
        logit(Yes) - logit(No), and with per_layer the read-out position's state after every layer."""
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
        if keep is not None:                      # drop some visual tokens; the rest keep their own positions
            sel = torch.ones(ids.shape[1], dtype=torch.bool, device=ids.device)
            sel[(ids[0] == self.image_token).nonzero()[:, 0]] = torch.cat([k.reshape(-1) for k in keep]).to(ids.device)
            embeds, mrope, present = embeds[:, sel], mrope[:, :, sel], present[:, sel]
        rope = lm.rotary_emb(embeds, mrope)
        keys = present.bool()[:, None, None, :]
        scan = enc.recurrent_mask(config=lm.config, inputs_embeds=embeds, attention_mask=present, past_key_values=None, position_ids=None)
        hidden, states = embeds, []
        for layer, softmax in enc.stack:
            hidden = layer(hidden, position_embeddings=rope, attention_mask=keys if softmax else scan, position_ids=None,
                           past_key_values=None, use_cache=False)
            if per_layer:
                states.append(hidden[0, -1])
        margin = self.readout(lm.norm(hidden)[:, -1])
        if not per_layer:
            return margin
        lens = [self.readout(lm.norm(s[None, None])[:, -1]) for s in states]
        return margin, torch.stack(states).float(), lens

    def readout(self, hidden) -> float:
        rt = self.rt
        logits = hidden.float() @ rt.readout.T if rt.readout_dtype == "float32" else torch.nn.functional.linear(hidden, rt.readout).float()
        logits = logits[0, :2] / rt.temperature
        return float(logits[0] - logits[1])


def pool_to(feature, grid, target):
    x = feature.view(grid[0], grid[1], -1).permute(2, 0, 1)[None].float()
    return torch.nn.functional.adaptive_avg_pool2d(x, target)


def run_pool(args) -> None:
    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.yaml").write_text(yaml.safe_dump({**config.raw, "intervention": {"mode": args.mode, "kernel": args.kernel, "factor": 2.0, "thr": args.thr, "dilate": args.dilate}}, sort_keys=False))
    backbone = BACKBONES.build(config.backbone); core = Core(backbone, config.question)
    items = list(DATASETS.build(config.dataset).items())
    n = 0
    with (args.out / "scores.jsonl").open("w") as stream, torch.inference_mode(), backbone.code.sdpa_backends(core.rt.device):
        for item in items:
            for window in plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames):
                images = read_frames(item.path, window.frame_indices); extra = {}
                big = core.encode(upsample(images, 2.0, args.kernel))
                feats = core.vision(big)
                if args.mode == "identity":
                    margin = core.language(big, feats)
                elif args.mode == "pool1x":
                    small = core.encode(images)
                    pooled = [pool_to(f, g2, g1)[0].permute(1, 2, 0).reshape(g1[0] * g1[1], -1).to(f.dtype)
                              for f, g2, g1 in zip(feats, big["grids"], small["grids"])]
                    margin = core.language(small, pooled)
                elif args.mode == "poolrep":
                    g1s = [(math.ceil(g[0] / 2), g[1] // 2) for g in big["grids"]]
                    rep = [torch.nn.functional.interpolate(pool_to(f, g2, g1), size=g2, mode="nearest")[0].permute(1, 2, 0)
                           .reshape(g2[0] * g2[1], -1).to(f.dtype) for f, g2, g1 in zip(feats, big["grids"], g1s)]
                    margin = core.language(big, rep)
                elif args.mode in ("mixrep", "mixdrop", "mixdrop_tight", "mixdrop_rand", "dropcoarse"):
                    # coarse tokens (2x2 block means) everywhere, fine 2x tokens only in the blocks of `fine`
                    g2 = big["grids"][0]; hb, wb = math.ceil(g2[0] / 2), math.ceil(g2[1] / 2)
                    m = cv2.resize(tgm_mask(images), (wb, hb), interpolation=cv2.INTER_AREA) > args.thr
                    fine = m if args.mode == "mixdrop_tight" else cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) if args.dilate else m
                    if args.mode == "dropcoarse":
                        fine = np.zeros_like(fine)
                    elif args.mode == "mixdrop_rand":        # as many fine blocks, at random places
                        r = np.random.default_rng(abs(hash((item.name, window.index))) % (2 ** 32))
                        flat = np.zeros(hb * wb, bool); flat[r.choice(hb * wb, int(fine.sum()), replace=False)] = True
                        fine = flat.reshape(hb, wb)
                    up = lambda t: t.repeat_interleave(2, 0).repeat_interleave(2, 1)[:g2[0], :g2[1]]
                    ft = up(torch.as_tensor(fine, device=feats[0].device))
                    anchor = torch.zeros(g2, dtype=torch.bool, device=feats[0].device); anchor[0::2, 0::2] = True
                    toks, keep = [], []
                    for f in feats:
                        x = f.view(g2[0], g2[1], -1)
                        c = torch.nn.functional.avg_pool2d(x.permute(2, 0, 1)[None].float(), 2, 2, ceil_mode=True, count_include_pad=False)[0]
                        rep = up(c.permute(1, 2, 0)).to(x.dtype)
                        toks.append(torch.where(ft[..., None], x, rep).reshape(g2[0] * g2[1], -1)); keep.append(ft | anchor)
                    extra = {"fine_blocks": int(fine.sum()), "blocks": int(hb * wb)}
                    if args.mode == "mixrep":
                        margin = core.language(big, toks); extra["tokens"] = int(len(feats) * g2[0] * g2[1])
                    else:
                        margin = core.language(big, toks, keep=keep); extra["tokens"] = int(sum(int(k.sum()) for k in keep))
                else:
                    raise ValueError(args.mode)
                p = 1 / (1 + math.exp(-margin))
                stream.write(json.dumps({"video": item.name, "window": window.index, "start_s": window.start_s,
                                         "end_s": window.end_s, "label": window.label, "score": p, "margin": margin, **extra}) + "\n")
                n += 1
                if n <= 3 and args.mode != "identity":
                    print(f"window {n}: {args.mode} p {p:.4f}   untouched 2x p {1 / (1 + math.exp(-core.language(big, feats))):.4f}", flush=True)
                if n % 2000 == 0:
                    stream.flush(); print(f"{n} windows", flush=True)
    (args.out / "run.json").write_text(json.dumps({"windows": n, "mode": args.mode, "kernel": args.kernel,
                                                   "finished": datetime.now(timezone.utc).isoformat()}))
    print("POOL_DONE", args.mode, args.kernel, n, flush=True)


def tgm_mask(images: list) -> np.ndarray:
    arrays = [np.asarray(im.convert("RGB")) for im in images]
    grey = [cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32) for a in arrays]
    heat = cv2.GaussianBlur(np.sum([np.abs(x - y) for x, y in zip(grey[1:], grey[:-1])], axis=0), (31, 31), 0)
    peak = float(heat.max())
    return (heat > 0.5 * peak).astype(np.float32) if peak > 0 else np.zeros_like(heat)


def run_layers(args) -> None:
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
    rng = np.random.default_rng(1)                       # a fresh draw, not the windows of the attention probe
    probe = []
    for g in ("small+", "large+", "near-", "far-", "normal-"):
        rows = sorted(groups[g]); idx = rng.choice(len(rows), min(args.per_group, len(rows)), replace=False)
        probe += [(g, *rows[i]) for i in idx]
    if args.smoke:
        probe = probe[:2]

    config = load_config(args.config)
    items = {key(item.name): item for item in DATASETS.build(config.dataset).items()}
    backbone = BACKBONES.build(config.backbone); core = Core(backbone, config.question)
    args.out.mkdir(parents=True, exist_ok=True)
    names = [("1x", 1.0, "bicubic"), ("bic", 2.0, "bicubic"), ("near", 2.0, "nearest")]
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)
    windows = {}
    with (args.out / "layers.jsonl").open("w") as stream, torch.inference_mode(), backbone.code.sdpa_backends(core.rt.device):
        for i, (g, v, w) in enumerate(probe):
            item = items[v]
            if v not in windows:
                windows[v] = {x.index: x for x in plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames)}
            images = read_frames(item.path, windows[v][w].frame_indices)
            mask = tgm_mask(images)
            R = {}
            for name, factor, kernel in names:
                inputs = core.encode(upsample(images, factor, kernel))
                feats = core.vision(inputs, blocks=True)
                blocks = [b.reshape(-1, b.shape[-1]) for b in core.block_out]
                margin, states, lens = core.language(inputs, feats, per_layer=True)
                R[name] = dict(inputs=inputs, feats=feats, blocks=blocks, margin=margin, states=states, lens=lens)
            if i == 0:
                ref = backbone.forward(ModelInput(images=images, text=None), config.question).logits
                print(f"check: runner {float(ref[0] - ref[1]):+.4f}  taken-apart path {R['1x']['margin']:+.4f}  vision blocks {len(R['1x']['blocks'])}  "
                      f"block tokens 1x {tuple(R['1x']['blocks'][0].shape)} 2x {tuple(R['bic']['blocks'][0].shape)}", flush=True)
            row = {"group": g, "video": v, "window": w, "margin": {n: R[n]["margin"] for n in R},
                   "lens": {n: R[n]["lens"] for n in R}}
            # vision tower, block by block
            row["vis_token_cos_bic_near"] = [float(cos(a, b).mean()) for a, b in zip(R["bic"]["blocks"], R["near"]["blocks"])]
            mean = lambda t: t.float().mean(0)
            for a, b in (("1x", "bic"), ("1x", "near"), ("bic", "near")):
                row[f"vis_mean_cos_{a}_{b}"] = [float(cos(mean(x), mean(y))) for x, y in zip(R[a]["blocks"], R[b]["blocks"])]
            # the tokens the language model receives: bicubic vs nearest, on and off the moving region; each vs 1x on the 1x grid
            g2, g1 = R["bic"]["inputs"]["grids"][0], R["1x"]["inputs"]["grids"][0]
            m2 = torch.as_tensor(cv2.resize(mask, (g2[1], g2[0]), interpolation=cv2.INTER_AREA).reshape(-1) > 0.25)
            c = torch.stack([cos(a, b) for a, b in zip(R["bic"]["feats"], R["near"]["feats"])]).cpu()     # [frames, tokens]
            row["merged_cos_bic_near"] = float(c.mean())
            row["merged_cos_bic_near_on"] = float(c[:, m2].mean()) if m2.any() else None
            row["merged_cos_bic_near_off"] = float(c[:, ~m2].mean()) if (~m2).any() else None
            for n in ("bic", "near"):
                p = [pool_to(f, g2, g1)[0].permute(1, 2, 0).reshape(g1[0] * g1[1], -1) for f in R[n]["feats"]]
                row[f"merged_cos_1x_{n}_pooled"] = float(torch.stack([cos(a, b) for a, b in zip(R["1x"]["feats"], p)]).mean())
            # language model: the read-out position's state, layer by layer
            for a, b in (("1x", "bic"), ("1x", "near"), ("bic", "near")):
                row[f"lm_cos_{a}_{b}"] = cos(R[a]["states"], R[b]["states"]).cpu().tolist()
            stream.write(json.dumps(row) + "\n"); stream.flush()
            if i % 25 == 0:
                print(f"layers {i}/{len(probe)}", flush=True)
    print("LAYERS_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool"); p.add_argument("--config", type=Path, required=True); p.add_argument("--mode", required=True)
    p.add_argument("--kernel", default="bicubic"); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--thr", type=float, default=0.05, help="a block is fine when the motion mask covers more than this share of it")
    p.add_argument("--dilate", type=int, default=1, help="1: also the eight blocks around every fine block")
    q = sub.add_parser("layers"); q.add_argument("--config", type=Path, required=True); q.add_argument("--events", type=Path, required=True)
    q.add_argument("--a", type=Path, required=True); q.add_argument("--b", type=Path); q.add_argument("--out", type=Path, required=True)
    q.add_argument("--per-group", type=int, default=60); q.add_argument("--cut", type=float, default=0.04); q.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    (run_pool if args.cmd == "pool" else run_layers)(args)
