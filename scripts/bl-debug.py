#!/usr/bin/env python3
"""Where does the sparse vision path leave the stock one? (2026-11-13) Stage-by-stage comparison on one frame."""
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "selftest", "--config", sys.argv[1], "--out", "/dev/null", "--limit", "-1"]
src = (Path(__file__).resolve().parent / "bl.py").read_text()
head = src[:src.index('if args.cmd == "selftest":')]
exec(compile(head, "bl.py", "exec"))

cap = {}
def pre(name):
    def f(mod, a, kw):
        cap.setdefault(name + ".in", (tuple(x.detach().clone() if torch.is_tensor(x) else x for x in a), {k: (v if not torch.is_tensor(v) else v.detach().clone()) for k, v in kw.items()}))
    return f
def post(name):
    def f(mod, a, kw, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        cap.setdefault(name + ".out", o.detach().clone() if torch.is_tensor(o) else o)
    return f
hs = [V.patch_embed.register_forward_hook(post("embed"), with_kwargs=True), V.blocks[0].register_forward_pre_hook(pre("b0"), with_kwargs=True),
      V.blocks[0].register_forward_hook(post("b0"), with_kwargs=True), V.blocks[-1].register_forward_hook(post("bl"), with_kwargs=True),
      V.merger.register_forward_hook(post("merger"), with_kwargs=True), V.rotary_pos_emb.register_forward_pre_hook(pre("rope"), with_kwargs=True)]
with torch.inference_mode(), backbone.code.sdpa_backends(DEV):
    item = items[0]; w = plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames)[3]
    images = read_frames(item.path, w.frame_indices); big = layout_for(images); g2 = big["grids"][0]
    frame = big_rgb(images[0], g2)
    cap.clear()
    pv, grid = stock([Image.fromarray(frame)])
    ref = core.model.get_image_features(pv, grid, return_dict=True).pooler_output[0]
    A = dict(cap); cap.clear()
    mine = sparse_vision([frame], [torch.ones(g2, dtype=torch.bool, device=DEV)], g2)[0]
    B = dict(cap)
    d = lambda x, y: f"max abs diff {float((x.float() - y.float()).abs().max()):.5f} (max abs {float(x.float().abs().max()):.2f})" if x.shape == y.shape else f"SHAPES {tuple(x.shape)} {tuple(y.shape)}"
    print("grid", grid.tolist(), "pv", tuple(pv.shape), pv.dtype, "V.dtype", V.dtype)
    print("patch embed out:", d(A["embed.out"], B["embed.out"]))
    a_in, b_in = A["b0.in"], B["b0.in"]
    print("block 0 input  :", d(a_in[0][0], b_in[0][0]))
    print("block 0 kwargs stock:", {k: (tuple(v.shape), str(v.dtype)) if torch.is_tensor(v) else (type(v).__name__ if isinstance(v, tuple) else v) for k, v in a_in[1].items()})
    print("block 0 kwargs mine :", {k: (tuple(v.shape), str(v.dtype)) if torch.is_tensor(v) else (type(v).__name__ if isinstance(v, tuple) else v) for k, v in b_in[1].items()})
    print("cu_seqlens stock", a_in[1]["cu_seqlens"].tolist(), "mine", b_in[1]["cu_seqlens"].tolist())
    pa, pb = a_in[1]["position_embeddings"], b_in[1]["position_embeddings"]
    print("rope cos:", d(pa[0], pb[0]), "| sin:", d(pa[1], pb[1]))
    ra, rb = A["rope.in"][0][1], B["rope.in"][0][1]
    print("rope position ids stock", tuple(ra.shape), ra[:6].tolist(), "mine", tuple(rb.shape), rb[:6].tolist(), "equal", bool((ra == rb).all()) if ra.shape == rb.shape else None)
    print("block 0 out    :", d(A["b0.out"], B["b0.out"]))
    print("last block out :", d(A["bl.out"], B["bl.out"]))
    print("merger out     :", d(A["merger.out"], B["merger.out"]))
    print("returned tokens:", d(ref, mine), "min cosine", float(F.cosine_similarity(ref.float(), mine.float(), dim=-1).min()))
with torch.inference_mode(), backbone.code.sdpa_backends(DEV):
    cosm = lambda x, y: (float(F.cosine_similarity(x.float(), y.float(), dim=-1).min()), float(F.cosine_similarity(x.float(), y.float(), dim=-1).mean()))
    print("bf16  sparse vs stock: min / mean cosine", cosm(ref, mine))
    for eps in (1e-4, 1e-3, 1e-2):
        noisy = core.model.get_image_features(pv + eps * torch.randn_like(pv), grid, return_dict=True).pooler_output[0]
        print(f"bf16  stock vs stock with input noise {eps:g}: min / mean cosine", cosm(ref, noisy))
    V.float()
    ref32 = V(pv.float(), grid_thw=grid, return_dict=True).pooler_output
    mine32 = sparse_vision([frame], [torch.ones(g2, dtype=torch.bool, device=DEV)], g2)[0]
    print("fp32  sparse vs stock: min / mean cosine", cosm(ref32, mine32), "max abs diff", float((ref32 - mine32).abs().max()))
    print("fp32 stock vs bf16 stock: min / mean cosine", cosm(ref32, ref))
print("DEBUG_DONE")
