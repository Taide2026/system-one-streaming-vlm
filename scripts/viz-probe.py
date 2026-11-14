#!/usr/bin/env python3
"""What changes inside the model when the frames are enlarged (2026-11-12). One GPU, the fine-tuned d3-nano.

For a window the script runs the same adapter on three inputs - original frames, all frames bicubic 2x, all frames
nearest 2x - through the runtime's own plain path (processor -> image embeddings -> PromptEncoder -> read-out at the
last position) and records:
  margin      logit(Yes) - logit(No) of the read-out
  attention   in each of the 6 softmax-attention layers, the attention of the last (read-out) position over the
              visual tokens (head mean), as one grid per frame; total share on visual tokens
  features    the vision tower's merged tokens of every frame (what the language model receives)
  occlusion   (examples only) drop of the margin when one cell of a 6 x 8 grid is blurred in all 8 frames

  python scripts/viz-probe.py --config configs/viz_floor.yaml --events results/pro6000/jev-hard-events.jsonl \
      --a <floor frames run> --b <floor all-2x run> --out results/nano4/viz/<time>
Writes probe.jsonl (one line per window and input: aggregate statistics) and examples/<id>.npz (+ frames as jpg).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import s1stream.plugins  # noqa: F401
from s1stream.backbones.d3 import build_questions
from s1stream.config import load_config
from s1stream.input_forms.frames import read_frames
from s1stream.input_forms.upsample import upsample
from s1stream.registry import BACKBONES, DATASETS
from s1stream.types import ModelInput
from s1stream.windows import plan_windows

INPUTS = {"1x": (1.0, "bicubic"), "2x_bicubic": (2.0, "bicubic"), "2x_nearest": (2.0, "nearest")}


def tgm_mask(images: list) -> np.ndarray:
    """The motion map of src/s1stream/input_forms/foveated.py: pixels above half the peak of the summed frame
    difference (float mask, original resolution)."""
    arrays = [np.asarray(im.convert("RGB")) for im in images]
    grey = [cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32) for a in arrays]
    heat = cv2.GaussianBlur(np.sum([np.abs(x - y) for x, y in zip(grey[1:], grey[:-1])], axis=0), (31, 31), 0)
    peak = float(heat.max())
    return (heat > 0.5 * peak).astype(np.float32) if peak > 0 else np.zeros_like(heat)


class Probe:
    def __init__(self, backbone, question) -> None:
        self.backbone, self.rt, self.question = backbone, backbone.model, question
        self.image_token = int(self.rt.backbone.config.image_token_id)
        assert self.rt.encoder is not None, "this checkpoint has no plain non-causal path"

    @torch.inference_mode()
    def run(self, images: list, capture: bool = True) -> dict:
        rt = self.rt
        prepared = rt.prepare(self.question.state, build_questions(self.question), list(images), None)
        key = prepared.runnable[0]
        encoded = rt.processor(text=[prepared.texts[key]], images=list(prepared.images), padding=True,
                               return_tensors="pt")
        inputs = {name: value.to(rt.device) for name, value in encoded.items()}
        ids = inputs["input_ids"]
        length = ids.shape[1]
        store: list = []
        original = torch.nn.functional.scaled_dot_product_attention

        def spy(q, k, v, *args, **kwargs):
            if capture and q.dim() == 4 and q.shape[0] == 1 and q.shape[-2] == length and k.shape[-2] == length:
                mask = kwargs.get("attn_mask", args[0] if args else None)
                ql, kk = q[:, :, -1:, :].float(), k.float()
                if kk.shape[1] != ql.shape[1]:
                    kk = kk.repeat_interleave(ql.shape[1] // kk.shape[1], dim=1)
                w = ql @ kk.transpose(-1, -2) / math.sqrt(ql.shape[-1])
                if mask is not None and mask.dtype == torch.bool:
                    w = w.masked_fill(~(mask[..., -1:, :]), float("-inf"))
                store.append(w.softmax(-1)[0, :, 0].cpu())          # [heads, L]
            return original(q, k, v, *args, **kwargs)

        backbone = rt.backbone
        with self.backbone.code.sdpa_backends(rt.device):
            embeds = backbone.get_input_embeddings()(ids)
            pooled = backbone.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"],
                                                 return_dict=True).pooler_output
            flat = torch.cat(pooled, dim=0).to(embeds.device, embeds.dtype)
            slots, _ = backbone.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=flat)
            embeds = embeds.masked_scatter(slots, flat)
            mrope = backbone.compute_3d_position_ids(
                input_ids=ids, image_grid_thw=inputs["image_grid_thw"], video_grid_thw=None, inputs_embeds=embeds,
                attention_mask=inputs["attention_mask"], past_key_values=None,
                mm_token_type_ids=inputs["mm_token_type_ids"])
            torch.nn.functional.scaled_dot_product_attention = spy
            try:
                hidden = rt.encoder(embeds, inputs["attention_mask"], mrope)[:, -1]
            finally:
                torch.nn.functional.scaled_dot_product_attention = original
        if rt.readout_dtype == "float32":
            logits = hidden.float() @ rt.readout.T
        else:
            logits = torch.nn.functional.linear(hidden, rt.readout).float()
        logits = logits[0, :2] / rt.temperature
        out = {"margin": float(logits[0] - logits[1]), "tokens": int((ids[0] == self.image_token).sum())}
        if not capture:
            return out
        grids = [(int(h) // 2, int(w) // 2) for _, h, w in inputs["image_grid_thw"].tolist()]
        positions = (ids[0] == self.image_token).nonzero()[:, 0].cpu()
        assert sum(h * w for h, w in grids) == len(positions), (grids, len(positions))
        out["grids"] = grids
        out["features"] = [p.float().cpu().numpy().astype(np.float16) for p in pooled]
        attention = torch.stack(store).float() if store else torch.zeros(0, 1, length)   # [layers, heads, L]
        out["layers"] = int(attention.shape[0])
        out["attention_visual_share"] = attention.mean(1)[:, positions].sum(-1).numpy()   # [layers]
        maps, start = [], 0
        mean = attention.mean(1)[:, positions].numpy()                                     # [layers, n visual]
        for h, w in grids:
            maps.append(mean[:, start:start + h * w].reshape(-1, h, w).astype(np.float32))
            start += h * w
        out["attention"] = maps                                                            # per frame [layers, h, w]
        return out


def redundancy(features: list, grids: list) -> tuple[float, float]:
    """Mean cosine similarity of horizontally / vertically adjacent tokens, and the effective rank (participation
    ratio of the singular values) of a frame's token matrix; mean over frames."""
    sims, ranks = [], []
    for f, (h, w) in zip(features, grids):
        x = f.astype(np.float32).reshape(h, w, -1)
        x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        sims.append(float(np.concatenate([(x[:, 1:] * x[:, :-1]).sum(-1).ravel(), (x[1:] * x[:-1]).sum(-1).ravel()]).mean()))
        c = f.astype(np.float32); c = c - c.mean(0)
        s = np.linalg.svd(c, compute_uv=False) ** 2
        ranks.append(float(s.sum() ** 2 / (s ** 2).sum()))
    return float(np.mean(sims)), float(np.mean(ranks))


def motion_stats(attention: list, grids: list, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Per layer: share of the visual attention that falls on the motion region, divided by the region's share of
    the frame (1 = no preference). Also the region's area share."""
    on = np.zeros(attention[0].shape[0]); total = np.zeros_like(on); area = []
    for a, (h, w) in zip(attention, grids):
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
        on += (a * m).sum((1, 2)); total += a.sum((1, 2)); area.append(float(m.mean()))
    share = float(np.mean(area))
    return (on / np.maximum(total, 1e-9)) / max(share, 1e-6), share


def occlusion(probe: Probe, images: list, factor: float, rows: int = 6, cols: int = 8) -> np.ndarray:
    base = probe.run(upsample(images, factor, "bicubic"), capture=False)["margin"]
    arrays = [np.asarray(im.convert("RGB")) for im in images]
    blurred = [cv2.GaussianBlur(a, (0, 0), 12) for a in arrays]
    height, width = arrays[0].shape[:2]
    out = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            y0, y1, x0, x1 = r * height // rows, (r + 1) * height // rows, c * width // cols, (c + 1) * width // cols
            occluded = []
            for a, b in zip(arrays, blurred):
                a = a.copy(); a[y0:y1, x0:x1] = b[y0:y1, x0:x1]; occluded.append(Image.fromarray(a))
            out[r, c] = base - probe.run(upsample(occluded, factor, "bicubic"), capture=False)["margin"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-group", type=int, default=40)
    ap.add_argument("--cut", type=float, default=0.04)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--occl-per-group", type=int, default=0, help="only: occlusion maps for this many probe windows per group")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "examples").mkdir(exist_ok=True)

    key = lambda n: n[:-4] if n.endswith(".mp4") else n
    events = [json.loads(l) for l in args.events.open()]
    by_video = defaultdict(list)
    for e in events:
        by_video[key(e["video"])].append((e["event"][0], e["event"][1], e["area_frac"]))
    load = lambda p: {(key(r["video"]), r["window"]): r for r in map(json.loads, (p / "scores.jsonl").open())}
    A, B = load(args.a), load(args.b)
    lg = lambda p: math.log(min(max(p, 1e-6), 1 - 1e-6) / (1 - min(max(p, 1e-6), 1 - 1e-6)))
    positives = defaultdict(list)
    for (v, w), r in A.items():
        if r["label"] == 1:
            positives[v].append(w)
    groups = defaultdict(list)
    for (v, w), r in A.items():
        d = lg(B[(v, w)]["score"]) - lg(r["score"])
        if r["label"] == 1:
            lo, hi = r["start_s"] * 30, r["end_s"] * 30 - 1
            area = next((a for s, e, a in by_video.get(v, []) if lo <= e and s <= hi), None)
            if area is not None:
                groups["small+" if area < args.cut else "large+"].append((v, w, d, area))
        elif v not in positives:
            groups["normal-"].append((v, w, d, None))
        else:
            groups["near-" if min(abs(w - p) for p in positives[v]) <= 5 else "far-"].append((v, w, d, None))
    rng = np.random.default_rng(0)
    probe_set = []
    for g in ("small+", "large+", "near-", "far-", "normal-"):
        rows = sorted(groups[g]); idx = rng.choice(len(rows), min(args.per_group, len(rows)), replace=False)
        probe_set += [(g, *rows[i]) for i in idx]

    def distinct(rows, n):
        seen, out = set(), []
        for row in rows:
            if row[0] not in seen:
                seen.add(row[0]); out.append(row)
            if len(out) == n:
                break
        return out
    examples = ([("small_gain", *r) for r in distinct(sorted(groups["small+"], key=lambda r: -r[2]), 6)]
                + [("large_flat", *r) for r in distinct(sorted(groups["large+"], key=lambda r: r[2]), 3)]
                + [("normal_rise", *r) for r in distinct(sorted(groups["normal-"], key=lambda r: -r[2]), 4)])
    if args.smoke:
        probe_set, examples = probe_set[:3], examples[:1]

    config = load_config(args.config)
    items = {key(item.name): item for item in DATASETS.build(config.dataset).items()}
    backbone = BACKBONES.build(config.backbone)
    probe = Probe(backbone, config.question)
    windows = {}

    def frames(v, w):
        item = items[v]
        if v not in windows:
            windows[v] = {x.index: x for x in plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames)}
        return read_frames(item.path, windows[v][w].frame_indices)

    # the plain path must reproduce the runner
    v, w = probe_set[0][1], probe_set[0][2]
    images = frames(v, w)
    ref = backbone.forward(ModelInput(images=images, text=None), config.question).logits
    mine = probe.run(images)
    print(f"check: runner margin {float(ref[0] - ref[1]):+.4f}  plain path {mine['margin']:+.4f}  "
          f"softmax layers seen {mine['layers']}  tokens {mine['tokens']}  recorded p {A[(v, w)]['score']:.4f}", flush=True)

    if args.occl_per_group:
        seen, maps = defaultdict(int), {}
        with (args.out / "occlusion.jsonl").open("w") as stream:
            for g, v, w, d, area in probe_set:
                if seen[g] >= args.occl_per_group:
                    continue
                seen[g] += 1
                images = frames(v, w); mask = tgm_mask(images)
                m = cv2.resize(mask, (8, 6), interpolation=cv2.INTER_AREA)
                row = {"group": g, "video": v, "window": w, "motion_area": float(m.mean())}
                for tag, factor in (("1x", 1.0), ("2x", 2.0)):
                    o = occlusion(probe, images, factor); maps[f"{g}__{v}__{w}__{tag}"] = o
                    pos = np.maximum(o, 0); peak = np.unravel_index(int(np.abs(o).argmax()), o.shape)
                    row[tag] = {"on_motion": float((pos * m).sum() / max(pos.sum(), 1e-9)), "abs_on_motion": float((np.abs(o) * m).sum() / max(np.abs(o).sum(), 1e-9)),
                                "peak_on_motion": float(m[peak]), "peak": float(o[peak]), "abs_sum": float(np.abs(o).sum()), "pos_sum": float(pos.sum())}
                maps[f"{g}__{v}__{w}__mask"] = m
                stream.write(json.dumps(row) + "\n"); stream.flush()
                print(f"occlusion {sum(seen.values())}", flush=True)
        np.savez_compressed(args.out / "occlusion_maps.npz", **maps)
        print("OCCL_DONE", flush=True)
        return

    with (args.out / "probe.jsonl").open("w") as stream:
        for i, (g, v, w, d, area) in enumerate(probe_set):
            images = frames(v, w); mask = tgm_mask(images)
            for name, (factor, kernel) in INPUTS.items():
                r = probe.run(upsample(images, factor, kernel))
                enrich, share = motion_stats(r["attention"], r["grids"], mask)
                sim, rank = redundancy(r["features"], r["grids"])
                per_frame = np.array([a.sum((1, 2)) for a in r["attention"]])            # [frames, layers]
                stream.write(json.dumps({
                    "group": g, "video": v, "window": w, "input": name, "margin": r["margin"], "tokens": r["tokens"],
                    "visual_share": r["attention_visual_share"].tolist(), "motion_enrichment": enrich.tolist(),
                    "motion_area": share, "neighbour_cosine": sim, "effective_rank": rank,
                    "frame_share": per_frame.tolist()}) + "\n")
            stream.flush()
            if i % 20 == 0:
                print(f"probe {i}/{len(probe_set)}", flush=True)

    for kind, v, w, d, area in examples:
        images = frames(v, w); mask = tgm_mask(images)
        name = f"{kind}__{v}__{w}"
        save = {"mask": mask, "delta_logit_recorded": d, "area": -1.0 if area is None else area,
                "p_a": A[(v, w)]["score"], "p_b": B[(v, w)]["score"], "start_s": A[(v, w)]["start_s"]}
        for tag, (factor, kernel) in INPUTS.items():
            r = probe.run(upsample(images, factor, kernel))
            save[f"{tag}__margin"] = r["margin"]; save[f"{tag}__grid"] = np.array(r["grids"][0])
            save[f"{tag}__attention"] = np.stack(r["attention"])                           # [frames, layers, h, w]
            save[f"{tag}__features"] = np.stack(r["features"])                             # [frames, h*w, D]
        save["occlusion_1x"] = occlusion(probe, images, 1.0)
        save["occlusion_2x"] = occlusion(probe, images, 2.0)
        np.savez_compressed(args.out / "examples" / f"{name}.npz", **save)
        for k, im in enumerate(images):
            im.save(args.out / "examples" / f"{name}__f{k}.jpg", quality=93)
        print(f"example {name}: recorded dlogit {d:+.2f}, margins 1x {save['1x__margin']:+.2f} "
              f"2x {save['2x_bicubic__margin']:+.2f} nearest {save['2x_nearest__margin']:+.2f}", flush=True)
    print("VIZ_DONE", flush=True)


if __name__ == "__main__":
    main()
