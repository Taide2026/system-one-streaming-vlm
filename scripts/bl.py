#!/usr/bin/env python3
"""Baselines for the key-frame + motion-crop layout, in one harness with it (2026-11-13).

Every mode keeps the key frames in full at 2x and differs in what the other frames contribute. The prompt layout is
that of an 8-frame 2x window; a mode decides which positions carry a token.

  crop        ours: the bounding box of the moving blocks, cut from the original frame, enlarged 2x, encoded as a small
              image of its own. --crop-scale 1 leaves the crop at the original scale (crop without enlarging).
              --hi <variant> (ComplexVAD): key frames and crops are REAL pixels of the 2x-resolution video, not bicubic.
  whole1x     the other frames whole at the original resolution (LoHi / Q-Frame style: few sharp frames + cheap frames)
  pool        every frame encoded at 2x, the other frames' tokens average-pooled 2x2 afterwards (SlowFast style)
  dtd         every frame encoded at 2x, of the other frames the tokens that changed most against the previous frame
              are kept, up to --budget tokens per window (differential token drop, feature level, TimeChat-Online / EVS style)
  patch-tgm   the other frames' patches are dropped BEFORE the vision tower; kept: the moving blocks of our own mask
              (same signal as crop, patch set instead of bounding box)
  patch-rect  ours, encoded in place (2026-11-13): the same rectangle as crop, but its patches go through the vision
              tower as part of the frame (their own position embeddings, sparse call) instead of as a separate image
  patch-boxes one small rectangle per group of moving blocks (2026-11-13 evening): the moving blocks are split into
              connected groups (8-neighbourhood) and each group's bounding box is kept, encoded in place. One group ->
              the single rectangle of patch-rect; several separate movers -> several small boxes, no space between them
  ctx         ours with context (2026-11-13): as crop, and every other frame is also sent whole at the original
              resolution (one coarse token per 2x2 block); the crop's fine tokens replace the coarse ones under it
  patch-diff  as patch-tgm, kept: the 2x2-token groups whose pixels changed most against the previous frame, up to
              --budget tokens per window, or all above --tau (Run-Length Tokenization / CodecSight style criterion)

Named baselines (2026-11-14), each with its own selection signal, on all eight frames enlarged 2x:
  fastv       FastV (ECCV 2024), rule of the official code (pkunlp-icler/FastV d165972, modeling_llama.py, in-place variant):
              the read-out position's attention, averaged over heads, ranks the visual tokens; the top --budget stay for
              all later layers. Official K = 2; here the first layer that has softmax attention (--fastv-layer 3).
              Official R = 50% -> --budget 1200.
  visionzip   VisionZip (CVPR 2025), rule of the official Qwen2.5-VL port (dvlab-research/VisionZip 8f86b55,
              Qwen2_5_VL/qwen2_5vl_visionzip.py): attention received in the LAST vision block (softmax, mean over heads,
              summed over queries, averaged over the 2x2 patches of a token) picks the dominant tokens; the rest are
              merged by key similarity into contextual tokens (target + mean of the tokens assigned to it), targets
              taken at a uniform step. Official shares 65% dominant + 5% contextual -> --budget 1680; other budgets keep 13:1.
  rlt         Run-Length Tokenization (NeurIPS 2024), rule of the official code (rccchoudhury/rlt 69ec8b4,
              static_token_utils.batched_find_idxs_to_keep): a 16x16 patch is static when its mean absolute difference to
              the previous frame (ImageNet-normalised pixels, mean over channels) is <= --tau (official 0.1); static
              patches are not encoded; the first frame is kept whole. Our tower merges 2x2 patches, so a token is kept
              when any of its four patches is. No length encoding (it needs training). Selection before the tower.
  codecsight  CodecSight (arXiv 2604.06036; no code released, OUR implementation of its patch selection): the window's
              frames are encoded as one H.264 group (x264, I P P ..., no B frames); a token group is dynamic when a
              motion vector over it is >= --mv-tau pixels (paper default 0.25) or the block is intra-coded; once dynamic it
              stays so until the next I frame (= the next window); the I frame is kept whole. Selection before the tower.

  --signal tgm (2026-11-14): the same three mechanisms with OUR motion map as the signal instead of their own, to separate
              the signal from the mechanism. fastv: the tokens are ranked by the motion heat our mask thresholds (one map per window, the same
              rank in every frame) instead of attention. visionzip: the dominant tokens are the top of the motion map; the
              contextual merge is unchanged. rlt / codecsight: a token is static when the motion map is below our usual
              threshold; with one map per window the two coincide (first frame whole + moving cells of the other seven).

  python scripts/bl.py eval    --config <yaml> --out <run dir> --mode <mode> [--key 0 7] [--budget 1015] [--tau 0.1]
  python scripts/bl.py latency --config <yaml> --out <json> --kernels ref|fla
  python scripts/bl.py selftest --config <yaml> --out /dev/null      sparse path == stock path when nothing is dropped
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MODES = ["crop", "whole1x", "pool", "dtd", "patch-tgm", "patch-diff", "patch-rect", "ctx", "patch-boxes", "fastv", "visionzip", "rlt", "codecsight"]
ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["eval", "latency", "selftest"])
ap.add_argument("--config", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--mode", default="crop", choices=MODES); ap.add_argument("--key", type=int, nargs="*", default=[0, 7])
ap.add_argument("--rest", default="touch", choices=["touch", "loose"]); ap.add_argument("--crop-scale", type=int, default=2, choices=[1, 2])
ap.add_argument("--budget", type=int, default=1015, help="visual tokens per window for dtd / patch-diff (key frames included)")
ap.add_argument("--tau", type=float, default=0.0, help="patch-diff: keep groups whose mean absolute pixel change exceeds tau (0: use --budget)")
ap.add_argument("--fastv-layer", type=int, default=3); ap.add_argument("--mv-tau", type=float, default=0.25)
ap.add_argument("--signal", default="own", choices=["own", "tgm"], help="fastv / visionzip / rlt / codecsight: their own signal, or our motion map")
ap.add_argument("--only", nargs="*", default=None, help="latency: row names to run (default all)")
ap.add_argument("--reuse-key", type=int, default=0, help="1: frame 0 is not encoded; its tokens are the previous window's frame-7 tokens (same video, consecutive window). Sparse modes only")
ap.add_argument("--hi", default="", help="crop mode: dataset variant holding the same frames at twice the resolution (real pixels)")
ap.add_argument("--kernels", default="ref", choices=["ref", "fla"]); ap.add_argument("--per-video", type=int, default=2)
ap.add_argument("--warmup", type=int, default=5); ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
if args.cmd == "latency" and args.kernels == "ref":
    sys.modules["fla"] = None

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech import Core, tgm_mask, upsample  # noqa: E402
from prune import forward as fastv_forward  # noqa: E402

import transformers.models.qwen3_5.modeling_qwen3_5 as M  # noqa: E402
from s1stream.config import load_config  # noqa: E402
from s1stream.input_forms.frames import read_frames  # noqa: E402
from s1stream.registry import BACKBONES, DATASETS  # noqa: E402
from s1stream.windows import plan_windows  # noqa: E402

config = load_config(args.config)
backbone = BACKBONES.build(config.backbone); core = Core(backbone, config.question)
ip = core.rt.processor.image_processor
ip.size = {"shortest_edge": 1024, "longest_edge": ip.size["longest_edge"]}     # let a small crop keep its size
items = list(DATASETS.build(config.dataset).items())
V = next(m for m in core.model.modules() if type(m).__name__ == "Qwen3_5VisionModel")
DEV = core.rt.device
MEAN = torch.tensor(ip.image_mean, device=DEV).view(3, 1, 1); STD = torch.tensor(ip.image_std, device=DEV).view(3, 1, 1)
LAYOUT, GRID, PREV = {}, {}, {}
RNG = np.random.default_rng(0)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
LAST_ATTN: dict = {}
V.blocks[-1].attn.register_forward_pre_hook(lambda m, a, kw: LAST_ATTN.update(hs=a[0] if a else kw["hidden_states"], cu=kw.get("cu_seqlens", a[1] if len(a) > 1 else None),
                                                                              pe=kw.get("position_embeddings")) if LAST_ATTN.get("on") else None, with_kwargs=True)


def tgm_scores(images: list, g2) -> torch.Tensor:
    """Our motion map on the 2x token grid, one value per token of every frame (prompt order)."""
    grey = [cv2.GaussianBlur(cv2.cvtColor(rgb(im), cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32) for im in images]
    heat = cv2.GaussianBlur(np.sum([np.abs(x - y) for x, y in zip(grey[1:], grey[:-1])], axis=0), (31, 31), 0)   # the map mech.tgm_mask thresholds
    m = cv2.resize(heat, (g2[1], g2[0]), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(m).to(DEV).flatten().repeat(len(images))


def visionzip_scores():
    """Per frame: attention received by every token in the last vision block, and its key (both averaged over the four
    patches of the token), as the official Qwen2.5-VL port computes them."""
    a = V.blocks[-1].attn; hs = LAST_ATTN["hs"]; L = hs.shape[0]
    q, k, _ = a.qkv(hs).reshape(L, 3, a.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = LAST_ATTN["pe"]
    q, k = M.apply_rotary_pos_emb_vision(q, k, cos, sin)
    cu = LAST_ATTN["cu"].tolist(); recv, keys = [], []
    for b, e in zip(cu[:-1], cu[1:]):
        qc, kc = q[b:e].transpose(0, 1).float(), k[b:e].transpose(0, 1).float()            # (heads, n, d)
        att = (qc @ kc.transpose(-1, -2) * a.scaling).softmax(-1)
        recv.append(att.mean(0).sum(0).view((e - b) // 4, 4).mean(-1))
        keys.append(kc.reshape(kc.shape[0], (e - b) // 4, 4, -1).mean(2).mean(0))
    return recv, keys


def codec_dynamic(images: list, g2) -> list:
    """CodecSight selection: the frames as one H.264 group; per P frame the token cells (16x16 original pixels = one
    macroblock) with a motion vector >= --mv-tau pixels or without any inter block (intra-coded), accumulated."""
    import io

    import av
    w, h = images[0].size; w2, h2 = w - w % 2, h - h % 2
    buf = io.BytesIO()
    with av.open(buf, "w", format="h264") as out:
        st = out.add_stream("libx264", rate=4); st.width, st.height, st.pix_fmt = w2, h2, "yuv420p"
        st.options = {"preset": "veryfast", "tune": "zerolatency", "bf": "0", "g": str(len(images)), "keyint_min": str(len(images)), "sc_threshold": "0"}
        for im in images:
            for pkt in st.encode(av.VideoFrame.from_image(im.convert("RGB").crop((0, 0, w2, h2)))):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    buf.seek(0); masks = []
    with av.open(buf, "r", format="h264") as inp:
        vs = inp.streams.video[0]; vs.codec_context.options = {"flags2": "+export_mvs"}
        acc = np.zeros(g2, bool)
        for frame in inp.decode(vs):
            if frame.key_frame or not masks:
                acc = np.zeros(g2, bool); masks.append(np.ones(g2, bool)); continue
            cover = np.zeros(g2, bool); dyn = np.zeros(g2, bool)
            sd = frame.side_data.get("MOTION_VECTORS")
            if sd is not None:
                mv = sd.to_ndarray()
                mag = np.hypot(mv["motion_x"], mv["motion_y"]) / np.maximum(mv["motion_scale"], 1)
                for x, y, bw, bh, m in zip(mv["dst_x"], mv["dst_y"], mv["w"], mv["h"], mag):
                    r0, r1 = max(int((y - bh / 2) // 16), 0), min(int((y + bh / 2 - 1) // 16), g2[0] - 1)
                    c0, c1 = max(int((x - bw / 2) // 16), 0), min(int((x + bw / 2 - 1) // 16), g2[1] - 1)
                    cover[r0:r1 + 1, c0:c1 + 1] = True
                    if m >= args.mv_tau:
                        dyn[r0:r1 + 1, c0:c1 + 1] = True
            acc = acc | dyn | ~cover                                                     # no inter block over the cell: intra-coded, treated as changed
            masks.append(acc.copy())
    assert len(masks) == len(images), (len(masks), len(images))
    return masks


def sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def layout_for(images: list) -> dict:
    size = images[0].size
    if size not in LAYOUT:
        LAYOUT[size] = core.encode(upsample(images, 2.0, "bicubic"))
    return LAYOUT[size]


def rgb(img) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def big_rgb(img, g2) -> np.ndarray:
    """The frame enlarged 2x (bicubic), at exactly the pixel size the 2x token grid stands for."""
    a = cv2.resize(rgb(img), (img.size[0] * 2, img.size[1] * 2), interpolation=cv2.INTER_CUBIC)
    if a.shape[:2] != (g2[0] * 32, g2[1] * 32):
        a = cv2.resize(a, (g2[1] * 32, g2[0] * 32), interpolation=cv2.INTER_CUBIC)
    return a


def grid_tables(g2):
    """Per 2x grid: learned-position interpolation rows, rotary position ids, and (h, w) -> row of the stock patch order."""
    if g2 not in GRID:
        thw = torch.tensor([[1, 2 * g2[0], 2 * g2[1]]], device=DEV)
        idx, wts = M.get_vision_interpolation_indices_and_weights(thw, num_grid_per_side=V.num_grid_per_side, mode=V.interpolation_mode,
                                                                  align_corners=V.interpolation_align_corners,
                                                                  spatial_merge_size=V.config.spatial_merge_size, kwargs={})
        pos = M.get_vision_position_ids(thw, V.spatial_merge_size, kwargs={})
        assert pos.shape[1] == 2, pos.shape                                                # (N, 2): h, w of every patch in stock order
        look = torch.full((2 * g2[0], 2 * g2[1]), -1, dtype=torch.long, device=DEV); look[pos[:, 0], pos[:, 1]] = torch.arange(pos.shape[0], device=DEV)
        cu, _ = M.get_vision_attention_seqlens(thw, V.config, kwargs={})
        GRID[g2] = (idx, wts, pos, look, cu.dtype)
    return GRID[g2]


def sparse_vision(frames: list, sel: list, g2) -> list:
    """Vision tower on the selected 2x2-token groups only. frames: HxWx3 uint8 at the 2x pixel size; sel: bool (g2) per
    frame. Returns per frame the merged tokens of its selected groups in raster order (None when nothing is selected)."""
    idx, wts, pos, look, cu_dtype = grid_tables(g2)
    gh, gw = g2
    px, rows, counts = [], [], []
    off = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=DEV)
    for a, s in zip(frames, sel):
        n = int(s.sum())
        counts.append(n)
        if n == 0:
            continue
        x = ((torch.from_numpy(a).to(DEV).permute(2, 0, 1).float() / 255 - MEAN) / STD).contiguous()
        p = x.reshape(3, gh, 2, 16, gw, 2, 16).permute(1, 4, 2, 5, 0, 3, 6)[s]                 # (n, 2, 2, 3, 16, 16)
        p = p.reshape(n * 4, 3, 1, 16, 16).expand(-1, -1, V.config.temporal_patch_size, -1, -1).reshape(n * 4, -1)
        rc = s.nonzero()                                                                     # (n, 2) raster order
        hw = (2 * rc[:, None, :] + off[None]).reshape(-1, 2)
        px.append(p); rows.append(look[hw[:, 0], hw[:, 1]])
    if not px:
        return [None] * len(frames)
    px, rows = torch.cat(px), torch.cat(rows)
    h = V.patch_embed(px.to(V.dtype))
    h = h + (V.pos_embed(idx[rows]) * wts[rows][:, :, None]).sum(1).to(h.dtype)
    rope = V.rotary_pos_emb(h, pos[rows])
    lens = [4 * c for c in counts if c]
    cu = torch.tensor([0] + list(np.cumsum(lens)), dtype=cu_dtype, device=DEV)
    for blk in V.blocks:
        h = blk(h, cu_seqlens=cu, max_seqlen=max(lens), position_embeddings=rope)
    merged = V.merger(h)
    out, at = [], 0
    for c in counts:
        out.append(merged[at:at + c] if c else None); at += c
    return out


def stock(send: list):
    enc = ip(images=send, return_tensors="pt")
    return enc["pixel_values"].to(DEV), enc["image_grid_thw"].to(DEV)


def one(images: list, mode: str, key: list, timed: bool = False, hi: list | None = None, prev7=None) -> dict:
    """One window. Returns margin, token and patch counts, stage times."""
    big = layout_for(images)
    g2 = big["grids"][0]; hb, wb = math.ceil(g2[0] / 2), math.ceil(g2[1] / 2)
    width, height = images[0].size
    others = [k for k in range(len(images)) if k not in key]
    t0 = sync() if timed else 0.0
    patches = 0
    keyimg = (lambda k: Image.fromarray(cv2.resize(rgb(hi[k]), (g2[1] * 32, g2[0] * 32), interpolation=cv2.INTER_AREA))) if hi is not None \
        else (lambda k: Image.fromarray(big_rgb(images[k], g2)))
    fine = None
    if mode in ("crop", "patch-tgm", "patch-rect", "ctx", "patch-boxes"):
        fine = cv2.resize(tgm_mask(images), (wb, hb), interpolation=cv2.INTER_AREA) > 0.05
        if args.rest == "loose":
            fine = cv2.dilate(fine.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    # ---- pre: what is sent ----
    rect, sel, send = None, None, None
    if mode in ("crop", "ctx"):
        send = [keyimg(k) for k in key]
        if mode == "ctx":
            send += [Image.fromarray(cv2.resize(rgb(images[k]), (wb * 32, hb * 32), interpolation=cv2.INTER_CUBIC)) for k in others]
        if fine.any() and others:
            rr, cc = np.where(fine); r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
            rect = (2 * r0, min(2 * (r1 + 1), g2[0]), 2 * c0, min(2 * (c1 + 1), g2[1]))
            ra, rb, ca, cb = rect
            x0, x1, y0, y1 = c0 * 32, min((c1 + 1) * 32, width), r0 * 32, min((r1 + 1) * 32, height)
            if args.crop_scale == 2:
                tw, th = (cb - ca) * 32, (rb - ra) * 32                                    # exact pixel size of the crop's token grid
            else:
                tw, th = math.ceil((cb - ca) / 2) * 32, math.ceil((rb - ra) / 2) * 32
            for k in others:
                if hi is not None:
                    fx, fy = hi[k].size[0] / width, hi[k].size[1] / height
                    a = rgb(hi[k])[round(y0 * fy):round(y1 * fy), round(x0 * fx):round(x1 * fx)]
                    send.append(Image.fromarray(cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA if a.shape[1] >= tw else cv2.INTER_CUBIC)))
                else:
                    send.append(Image.fromarray(cv2.resize(rgb(images[k])[y0:y1, x0:x1], (tw, th), interpolation=cv2.INTER_CUBIC)))
    elif mode == "whole1x":
        send = [keyimg(k) for k in key] + [Image.fromarray(cv2.resize(rgb(images[k]), (wb * 32, hb * 32), interpolation=cv2.INTER_CUBIC)) for k in others]
    elif mode in ("pool", "dtd", "fastv", "visionzip"):
        send = [Image.fromarray(big_rgb(im, g2)) for im in images]
    elif mode in ("rlt", "codecsight"):
        frames = [big_rgb(im, g2) for im in images]
        if args.signal == "tgm":
            hb_, wb_ = math.ceil(g2[0] / 2), math.ceil(g2[1] / 2)
            cellm = cv2.resize(tgm_mask(images), (wb_, hb_), interpolation=cv2.INTER_AREA) > 0.05
            cell = torch.as_tensor(cellm, device=DEV).repeat_interleave(2, 0).repeat_interleave(2, 1)[:g2[0], :g2[1]]
            sel = [torch.ones(g2, dtype=torch.bool, device=DEV)] + [cell] * (len(images) - 1)
        elif mode == "rlt":
            x = torch.from_numpy(np.stack(frames)).to(DEV).permute(0, 3, 1, 2).float() / 255 / IMNET_STD
            d = F.avg_pool2d((x[1:] - x[:-1]).abs(), 16).mean(1)                              # (T-1, 2*g2): per 16x16 patch
            moved = F.max_pool2d((d > args.tau).float()[:, None], 2)[:, 0].bool()           # a token is kept when any of its patches is
            sel = [torch.ones(g2, dtype=torch.bool, device=DEV)] + [moved[k] for k in range(len(images) - 1)]
        else:
            sel = [torch.as_tensor(m, device=DEV) for m in codec_dynamic(images, g2)]
        frames = [a if bool(s.any()) else None for a, s in zip(frames, sel)]
    else:                                                                                   # patch-tgm / patch-diff
        if mode in ("patch-tgm", "patch-rect", "patch-boxes"):
            if mode == "patch-rect" and fine.any():
                rr, cc = np.where(fine); fine = np.zeros_like(fine); fine[rr.min():rr.max() + 1, cc.min():cc.max() + 1] = True
            if mode == "patch-boxes" and fine.any():
                n_lab, _, stats, _ = cv2.connectedComponentsWithStats(fine.astype(np.uint8), connectivity=8)
                fine = np.zeros_like(fine)
                for x_, y_, w_, h_, _a in stats[1:n_lab]:
                    fine[y_:y_ + h_, x_:x_ + w_] = True
            cell = torch.as_tensor(fine, device=DEV).repeat_interleave(2, 0).repeat_interleave(2, 1)[:g2[0], :g2[1]]
            sel = [torch.ones(g2, dtype=torch.bool, device=DEV) if k in key else cell for k in range(len(images))]
        else:
            gray = torch.stack([torch.from_numpy(cv2.cvtColor(rgb(im), cv2.COLOR_RGB2GRAY)) for im in images]).to(DEV).float() / 255
            d = (gray[1:] - gray[:-1]).abs()[:, None]
            d = F.adaptive_avg_pool2d(d, g2)[:, 0]                                         # (T-1, g2): change against the previous frame
            score = torch.full((len(images), *g2), -1.0, device=DEV)
            for k in others:
                score[k] = d[k - 1] if k > 0 else d[0]
            if args.tau > 0:
                keepm = score > args.tau
            else:
                n = max(args.budget - len(key) * g2[0] * g2[1], 0)
                keepm = torch.zeros_like(score, dtype=torch.bool)
                if n:
                    flat = score.flatten(); top = flat.topk(min(n, int((flat >= 0).sum()))).indices
                    keepm.view(-1)[top] = True
            sel = [torch.ones(g2, dtype=torch.bool, device=DEV) if k in key else keepm[k] for k in range(len(images))]
        if prev7 is not None and 0 in key:
            sel[0] = torch.zeros(g2, dtype=torch.bool, device=DEV)                          # reused from the previous window: not encoded
        frames = [big_rgb(im, g2) if bool(s.any()) else None for im, s in zip(images, sel)]
    t1 = sync() if timed else 0.0
    # ---- proc + vision ----
    if sel is None:
        pixel_values, grid = stock(send)
        t2 = sync() if timed else 0.0
        LAST_ATTN["on"] = mode == "visionzip"
        feats = list(core.model.get_image_features(pixel_values, grid, return_dict=True).pooler_output)
        LAST_ATTN["on"] = False
        patches = int(grid.prod(-1).sum())
    else:
        t2 = t1
        feats = sparse_vision(frames, sel, g2)
        patches = int(sum(4 * int(s.sum()) for s in sel))
    t3 = sync() if timed else 0.0
    # ---- place the tokens in the 2x layout ----
    ref = next(f for f in feats if f is not None)
    dim, dt = ref.shape[-1], ref.dtype
    toks = [torch.zeros(g2[0] * g2[1], dim, device=DEV, dtype=dt) for _ in images]
    keep = [torch.zeros(g2, dtype=torch.bool, device=DEV) for _ in images]
    if sel is not None:
        for k, (f, s) in enumerate(zip(feats, sel)):
            if f is not None:
                toks[k][s.flatten()] = f; keep[k] = s.clone()
        if prev7 is not None and 0 in key:
            toks[0] = prev7.clone(); keep[0][:] = True
    elif mode == "fastv":
        t4 = sync() if timed else 0.0
        if args.signal == "tgm":
            margin, kept = fastv_forward(core, big, feats, args.budget, "given", args.fastv_layer, RNG, given=tgm_scores(images, g2))
        else:
            margin, kept = fastv_forward(core, big, feats, args.budget, "attention", args.fastv_layer, RNG)
        t5 = sync() if timed else 0.0
        return {"margin": margin, "tokens": int(kept), "patches": patches, "last": None, "pre": t1 - t0, "proc": t2 - t1, "vision": t3 - t2, "tokens_t": t4 - t3, "lm": t5 - t4}
    elif mode == "visionzip":
        recv, keys = visionzip_scores()
        score = torch.cat(recv); metric = torch.cat(keys)
        if args.signal == "tgm":
            t_ = tgm_scores(images, g2); score = t_ / t_.max().clamp(min=1e-6) + 1e-6 * score / score.max()                   # motion map decides; attention only breaks ties
        hidden = torch.cat(feats).float(); N = score.shape[0]
        n_dom = min(int(round(args.budget * 0.65 / 0.70)), N); n_ctx = max(args.budget - n_dom, 1)
        dom = torch.zeros(N, dtype=torch.bool, device=DEV); dom[score.topk(n_dom).indices] = True
        rest = (~dom).nonzero()[:, 0]
        out = hidden.clone(); chosen = dom.clone()
        if len(rest):
            mn = metric[rest]; mn = mn / mn.norm(dim=-1, keepdim=True)
            step = max(1, len(rest) // n_ctx); tgt = torch.arange(0, len(rest), step, device=DEV)[:n_ctx]
            is_t = torch.zeros(len(rest), dtype=torch.bool, device=DEV); is_t[tgt] = True
            assign = (mn[~is_t] @ mn[tgt].T).argmax(-1)
            agg = torch.zeros(len(tgt), hidden.shape[-1], device=DEV).index_add_(0, assign, hidden[rest[~is_t]])
            cnt = torch.zeros(len(tgt), device=DEV).index_add_(0, assign, torch.ones(len(assign), device=DEV)).clamp(min=1)
            out[rest[tgt]] = hidden[rest[tgt]] + agg / cnt[:, None]
            chosen[rest[tgt]] = True
        per = g2[0] * g2[1]
        for k in range(len(images)):
            toks[k] = out[k * per:(k + 1) * per].to(dt); keep[k] = chosen[k * per:(k + 1) * per].view(g2)
    elif mode in ("pool", "dtd"):
        full = torch.stack([f.view(g2[0], g2[1], dim) for f in feats])                      # (T, h, w, d)
        for k in key:
            toks[k] = feats[k].clone(); keep[k][:] = True
        if mode == "pool":
            for k in others:
                c = F.avg_pool2d(full[k].permute(2, 0, 1)[None].float(), 2, 2, ceil_mode=True, count_include_pad=False)[0].permute(1, 2, 0).to(dt)
                toks[k].view(g2[0], g2[1], dim)[0::2, 0::2] = c; keep[k][0::2, 0::2] = True
        else:
            change = torch.full((len(images), *g2), -1.0, device=DEV)
            for k in others:
                change[k] = 1 - F.cosine_similarity(full[k].float(), full[k - 1 if k > 0 else 1].float(), dim=-1)
            n = max(args.budget - len(key) * g2[0] * g2[1], 0)
            if n:
                top = change.flatten().topk(min(n, len(others) * g2[0] * g2[1])).indices
                km = torch.zeros_like(change, dtype=torch.bool); km.view(-1)[top] = True
                for k in others:
                    toks[k][km[k].flatten()] = feats[k][km[k].flatten()]; keep[k] = km[k]
    else:
        for i, k in enumerate(key):
            assert feats[i].shape[0] == g2[0] * g2[1], (feats[i].shape, g2)
            toks[k] = feats[i].clone(); keep[k][:] = True
        n_ctx = len(others) if mode in ("whole1x", "ctx") else 0
        if n_ctx:
            for i, k in enumerate(others):
                f = feats[len(key) + i]; assert f.shape[0] == hb * wb, (f.shape, hb, wb)
                t = toks[k].view(g2[0], g2[1], dim)
                t[0::2, 0::2] = f.view(hb, wb, dim)[:t[0::2, 0::2].shape[0], :t[0::2, 0::2].shape[1]]; keep[k][0::2, 0::2] = True
        if mode != "whole1x" and rect is not None:
            ra, rb, ca, cb = rect
            for i, k in enumerate(others):
                f = feats[len(key) + n_ctx + i]; t = toks[k].view(g2[0], g2[1], dim)
                if args.crop_scale == 2:
                    assert f.shape[0] == (rb - ra) * (cb - ca), (f.shape, rect)
                    t[ra:rb, ca:cb] = f.view(rb - ra, cb - ca, dim); keep[k][ra:rb, ca:cb] = True
                else:
                    hh, ww = math.ceil((rb - ra) / 2), math.ceil((cb - ca) / 2)
                    assert f.shape[0] == hh * ww, (f.shape, rect)
                    t[ra:rb:2, ca:cb:2] = f.view(hh, ww, dim); keep[k][ra:rb:2, ca:cb:2] = True
    t4 = sync() if timed else 0.0
    margin = core.language(big, toks, keep=keep)
    t5 = sync() if timed else 0.0
    return {"margin": margin, "tokens": int(sum(int(k.sum()) for k in keep)), "patches": patches, "last": toks[len(images) - 1].detach() if bool(keep[len(images) - 1].all()) else None,
            "pre": t1 - t0, "proc": t2 - t1, "vision": t3 - t2, "tokens_t": t4 - t3, "lm": t5 - t4}


def hi_frames(item, window):
    if not args.hi:
        return None
    variant = config.raw["dataset"]["variant"]
    return read_frames(Path(str(item.path).replace(f"/{variant}/", f"/{args.hi}/")), window.frame_indices)


if args.cmd == "selftest":
    with torch.inference_mode(), backbone.code.sdpa_backends(DEV):
        item = items[0]; w = plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames)[3]
        images = read_frames(item.path, w.frame_indices); big = layout_for(images); g2 = big["grids"][0]
        frames = [big_rgb(im, g2) for im in images]
        pv, grid = stock([Image.fromarray(a) for a in frames])
        ref = list(core.model.get_image_features(pv, grid, return_dict=True).pooler_output)
        full = [torch.ones(g2, dtype=torch.bool, device=DEV) for _ in images]
        mine = sparse_vision(frames, full, g2)
        err = max(float((a.float() - b.float()).abs().max()) for a, b in zip(ref, mine)); scale = float(ref[0].float().abs().mean())
        cosf = float(np.mean([float(F.cosine_similarity(a.float(), b.float(), dim=-1).mean()) for a, b in zip(ref, mine)]))   # bf16: per-token minimum is noise (fp32 gives 1.0000)
        x0 = ((torch.from_numpy(frames[0]).to(DEV).permute(2, 0, 1).float() / 255 - MEAN) / STD).reshape(3, g2[0], 2, 16, g2[1], 2, 16).permute(1, 4, 2, 5, 0, 3, 6)
        x0 = x0.reshape(-1, 3, 1, 16, 16).expand(-1, -1, 2, -1, -1).reshape(4 * g2[0] * g2[1], -1)
        pix = float((x0 - pv[:x0.shape[0]].float()).abs().max())
        twice = list(core.model.get_image_features(pv, grid, return_dict=True).pooler_output)
        print(f"pixel patches, mine against the processor: max abs difference {pix:.5f}; stock path run twice: max abs difference {max(float((a.float() - b.float()).abs().max()) for a, b in zip(ref, twice)):.4f}; "
              f"largest feature {float(ref[0].float().abs().max()):.1f}; mean per-token cosine sparse vs stock {cosf:.5f}")
        print(f"sparse path, nothing dropped: max abs difference {err:.4f} (mean abs feature {scale:.4f}); grid {g2}, pixel grid {grid[0].tolist()}")
        half = [s.clone() for s in full]
        for s in half[1:]:
            s[:, g2[1] // 2:] = False
        part = sparse_vision(frames, half, g2)
        cos = float(F.cosine_similarity(part[1].float(), ref[1].view(g2[0], g2[1], -1)[:, :g2[1] // 2].reshape(part[1].shape).float(), dim=-1).mean())
        print(f"half the patches dropped: {part[1].shape[0]} tokens, mean cosine to the full-frame tokens {cos:.3f}")
        for mode in MODES:
            r = one(images, mode, args.key, timed=True)
            print(f"{mode:10s} margin {r['margin']:+.3f} tokens {r['tokens']} patches {r['patches']}")
        whole = core.language(big, ref)
        for mode, over in (("fastv", {"budget": 10 ** 6}), ("rlt", {"tau": -1.0}), ("codecsight", {"mv_tau": -1.0})):   # nothing dropped: must give the whole-2x margin
            saved = {k: getattr(args, k) for k in over}
            for k, v in over.items():
                setattr(args, k, v)
            r = one(images, mode, args.key)
            for k, v in saved.items():
                setattr(args, k, v)
            print(f"{mode:10s} nothing dropped: margin {r['margin']:+.4f} tokens {r['tokens']}; whole 2x {whole:+.4f}")
        for name, mode, over in (("fastv 1200", "fastv", {"budget": 1200}), ("visionzip 1680", "visionzip", {"budget": 1680}), ("visionzip 1015", "visionzip", {"budget": 1015}),
                                 ("rlt .1", "rlt", {"tau": 0.1}), ("codecsight .25", "codecsight", {})):
            saved = {k: getattr(args, k) for k in over}
            for k, v in over.items():
                setattr(args, k, v)
            for it in items[:6]:
                ws = plan_windows(it, config.window.length_s, config.window.stride_s, config.window.frames)
                r = one(read_frames(it.path, ws[min(3, len(ws) - 1)].frame_indices), mode, args.key)
                print(f"{name:15s} {it.name[:24]:24s} margin {r['margin']:+.3f} tokens {r['tokens']} patches {r['patches']}")
            for k, v in saved.items():
                setattr(args, k, v)
        args.signal = "tgm"
        for name, mode, over in (("fastv 1015 tgm", "fastv", {"budget": 1015}), ("visionzip 1015 tgm", "visionzip", {"budget": 1015}), ("rlt tgm", "rlt", {})):
            saved = {k: getattr(args, k) for k in over}
            for k, v in over.items():
                setattr(args, k, v)
            r = one(images, mode, args.key); print(f"{name:18s} margin {r['margin']:+.3f} tokens {r['tokens']} patches {r['patches']}")
            for k, v in saved.items():
                setattr(args, k, v)
        args.signal = "own"
        print("SELFTEST_OK" if cosf > 0.99 else "SELFTEST_MISMATCH")
elif args.cmd == "eval":
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.yaml").write_text(yaml.safe_dump({**config.raw, "bl": {"mode": args.mode, "key": args.key, "rest": args.rest, "crop_scale": args.crop_scale,
                                                                              "budget": args.budget, "tau": args.tau, "hi": args.hi, "reuse_key": args.reuse_key,
                                                                              "fastv_layer": args.fastv_layer, "mv_tau": args.mv_tau, "signal": args.signal}}, sort_keys=False, default_flow_style=None))
    n = 0
    with (args.out / "scores.jsonl").open("w") as stream, torch.inference_mode(), backbone.code.sdpa_backends(DEV):
        for item in items:
            for window in plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames):
                reuse = None
                if args.reuse_key and PREV.get("video") == item.name and PREV.get("index") == window.index - 1 and PREV.get("last") is not None:
                    reuse = PREV["last"]
                r = one(read_frames(item.path, window.frame_indices), args.mode, args.key, hi=hi_frames(item, window), prev7=reuse)
                PREV.update(video=item.name, index=window.index, last=r.get("last"))
                stream.write(json.dumps({"video": item.name, "window": window.index, "start_s": window.start_s, "end_s": window.end_s,
                                         "label": window.label, "score": 1 / (1 + math.exp(-r["margin"])), "margin": r["margin"],
                                         "tokens": r["tokens"], "patches": r["patches"]}) + "\n")
                n += 1
                if n % 2000 == 0:
                    stream.flush(); print(f"{n} windows", flush=True)
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break
    (args.out / "run.json").write_text(json.dumps({"windows": n, "finished": datetime.now(timezone.utc).isoformat()}))
    print("BL_DONE", n, flush=True)
else:
    plan = [(item, w) for item in items for w in plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames)[:args.per_video]]
    if args.limit:
        plan = plan[:args.limit]
    try:
        import fla  # noqa: F401
        kern = "fla"
    except Exception:
        kern = "reference"
    result = {"gpu": torch.cuda.get_device_name(), "kernels": kern, "windows": len(plan) - args.warmup, "rows": {}}
    stages = ("decode", "pre", "proc", "vision", "tokens_t", "lm", "total")
    with torch.inference_mode(), backbone.code.sdpa_backends(DEV):
        for name, mode, over in (("crop (ours)", "crop", {}), ("crop, not enlarged", "crop", {"crop_scale": 1}), ("whole 1x rest", "whole1x", {}),
                                 ("pooled rest", "pool", {}), ("dtd 1015", "dtd", {}), ("patch tgm", "patch-tgm", {}),
                                 ("patch diff 1015", "patch-diff", {}), ("patch diff tau .02", "patch-diff", {"tau": 0.02}),
                                 ("rect in place", "patch-rect", {}), ("crop + coarse context", "ctx", {}), ("boxes in place", "patch-boxes", {}),
                                 ("fastv 1200", "fastv", {"budget": 1200}), ("fastv 1015", "fastv", {"budget": 1015}),
                                 ("visionzip 1680", "visionzip", {"budget": 1680}), ("visionzip 1015", "visionzip", {"budget": 1015}),
                                 ("rlt tau .1", "rlt", {"tau": 0.1}), ("codecsight .25", "codecsight", {})):
            if args.only and name not in args.only:
                continue
            saved = {k: getattr(args, k) for k in over}
            for k, v in over.items():
                setattr(args, k, v)
            rows = []
            for item, w in plan:
                t0 = sync(); images = read_frames(item.path, w.frame_indices); t1 = sync()
                r = one(images, mode, args.key, timed=True); r["decode"] = t1 - t0
                r["total"] = r["decode"] + r["pre"] + r["proc"] + r["vision"] + r["tokens_t"] + r["lm"]; rows.append(r)
            for k, v in saved.items():
                setattr(args, k, v)
            rows = rows[args.warmup:]
            out = {s: {"p50": 1000 * float(np.percentile([x[s] for x in rows], 50)), "p95": 1000 * float(np.percentile([x[s] for x in rows], 95)),
                       "mean": 1000 * float(np.mean([x[s] for x in rows]))} for s in stages}
            out["n_tokens"] = float(np.mean([x["tokens"] for x in rows])); out["n_patches"] = float(np.mean([x["patches"] for x in rows]))
            result["rows"][name] = out
            print(f"{kern:9s} {name:20s} tokens {out['n_tokens']:6.0f} patches {out['n_patches']:6.0f} | total p50 {out['total']['p50']:5.0f} p95 {out['total']['p95']:5.0f} ms | mean ms: "
                  + "  ".join(f"{s} {out[s]['mean']:.0f}" for s in stages[:-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print("BL_LAT_DONE", flush=True)
