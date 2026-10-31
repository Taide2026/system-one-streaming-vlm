"""Find which difference between the vit-1 path and the encoder-extract path moves the qwen35-vit features.

vit-1 ran the ViT inside the d3-flash runtime (D3.from_pretrained). encoder-extract runs Qwen3_5VisionModel alone.
Checks, on test windows (the 3 with the lowest patch_max cosine against vit-1, plus windows 0 and 1):
  1. vision config: Qwen3_5VisionModel from the d3-flash vision_config vs from the qwen3.5-9b vision_config.
  2. pixel_values: cast to bfloat16 before the call (encoder-extract) vs float32 into the call (the model casts).
  3. processor size: the d3 runtime sets shortest_edge 65,536 and longest_edge 1,638,400 (d3_runtime.load_processor).
  4. patch embedding: the d3 runtime replaces the Conv3d forward by F.linear (d3_runtime.linearize_patch_embed),
     and runs the ViT under d3_runtime.sdpa_backends.
Each line prints the max abs difference of the 28 x 1152 patch_mean and patch_max (0 = bitwise equal), and the
min cosine against vit-1.

Usage (worktree root): uv run python results/spark/encoder_probe_ucf_short_v1_f8/20261028-five-encoders/scripts/verify_pipeline.py
"""
import copy
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, Qwen3_5VisionModel

import s1stream.plugins  # noqa: F401  Register the plugins.
from s1stream.backbones.d3 import load_release_code
from s1stream.encoder.adapters import load_encoder
from s1stream.encoder.extract import load_encoder_config
from s1stream.encoder.tap import StateTap, pool
from s1stream.probe.extract import plan_sets
from s1stream.registry import DATASETS, INPUT_FORMS
from s1stream.runner import free_memory_gb

CONFIG = Path("configs/encoder_probe_ucf_short_v1_f8.yaml")
assets = Path(os.environ["S1_ASSETS"])
features = Path(os.environ["S1_FEATURES"])
old = features / "probe_ucf_short_v1_f8" / "V" / "vit-1" / "test"
new = features / "encoder_probe_ucf_short_v1_f8" / "qwen35-vit" / "run-1" / "test"

memory = free_memory_gb()
if memory is not None and memory < 4 + 10:
    raise SystemExit(f"Insufficient available memory: {memory:.1f} GiB")
torch.cuda.set_per_process_memory_fraction(8 / (torch.cuda.get_device_properties(0).total_memory / 2**30))

old_max = np.load(old / "last.npy", mmap_mode="r")
new_max = np.load(new / "patch_max.npy", mmap_mode="r")
cos = (np.sum(old_max * new_max, -1) / np.linalg.norm(old_max, axis=-1) / np.linalg.norm(new_max, axis=-1)).min(1)
windows = sorted(set(np.argsort(cos)[:3].tolist()) | {0, 1})
print("windows", windows, "min patch_max cosine run-1 vs vit-1", [round(float(cos[i]), 6) for i in windows])

config = load_encoder_config(CONFIG)
rows = plan_sets(DATASETS.build(config.train).items(), DATASETS.build(config.test).items(), config.window)["test"]
input_form = INPUT_FORMS.build(config.input_form)
frames = {i: input_form.build(rows[i].item, rows[i].window, None).images for i in windows}

adapter = load_encoder("qwen35-vit", assets / "qwen3.5-9b", "cuda")
qwen_model, qwen_processor = adapter.model, adapter.processor
print("attn_implementation (encoder-extract):", qwen_model.config._attn_implementation)

# 1. vision config
qwen_cfg = AutoConfig.from_pretrained(assets / "qwen3.5-9b").vision_config
d3_cfg = AutoConfig.from_pretrained(assets / "d3-flash").vision_config
a, b = qwen_cfg.to_dict(), d3_cfg.to_dict()
print("vision config keys that differ:", {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)})
d3_model = Qwen3_5VisionModel._from_config(d3_cfg, torch_dtype=torch.bfloat16)
d3_model.load_state_dict(qwen_model.state_dict(), strict=True)
d3_model = d3_model.to("cuda").eval()
inv_a, inv_b = qwen_model.rotary_pos_emb.inv_freq, d3_model.rotary_pos_emb.inv_freq
print("rotary inv_freq equal:", torch.equal(inv_a, inv_b))

# 4. the d3 runtime's patch embedding and attention setting
code = load_release_code(assets / "d3-flash")
d3_linear = copy.deepcopy(d3_model)
print("patch embeddings linearized:", code.linearize_patch_embed(d3_linear))

d3_processor = AutoImageProcessor.from_pretrained(assets / "d3-flash")
d3_processor.size = {"shortest_edge": code.IMAGE_MIN_PIXELS, "longest_edge": code.IMAGE_MAX_PIXELS}


def run(model, processor, images, cast_first, context=None):
    encoded = processor(images=images, return_tensors="pt")
    pixels = encoded["pixel_values"].to("cuda")
    if cast_first:
        pixels = pixels.to(torch.bfloat16)
    with torch.inference_mode(), (context or torch.inference_mode()), StateTap(model.blocks) as tap:
        model(pixels, grid_thw=encoded["image_grid_thw"].to("cuda"))
        kinds = pool(tap.states(), prefix=0, cls=False)
    return {k: v.numpy() for k, v in kinds.items()}, encoded


def cosine(x, y):
    return float((np.sum(x * y, -1) / np.linalg.norm(x, axis=-1) / np.linalg.norm(y, axis=-1)).min())


paths = {
    "A encoder-extract (qwen config, bf16 first)": lambda im: run(qwen_model, qwen_processor, im, True),
    "B qwen config, float32 into the call": lambda im: run(qwen_model, qwen_processor, im, False),
    "C d3 config, float32 into the call": lambda im: run(d3_model, qwen_processor, im, False),
    "D C + d3 processor size": lambda im: run(d3_model, d3_processor, im, False),
    "E D + linear patch embedding": lambda im: run(d3_linear, d3_processor, im, False),
    "F E + d3 sdpa_backends (= vit-1 path)": lambda im: run(d3_linear, d3_processor, im, False,
                                                           code.sdpa_backends("cuda")),
}
for i in windows:
    vit1 = {"patch_mean": np.load(old / "visual.npy", mmap_mode="r")[i], "patch_max": old_max[i]}
    stored = {"patch_mean": np.load(new / "patch_mean.npy", mmap_mode="r")[i], "patch_max": new_max[i]}
    results = {name: f(frames[i]) for name, f in paths.items()}
    base, base_enc = results[next(iter(paths))]
    print(f"\nwindow {i} ({rows[i].item.name} w{rows[i].window.index}), grid_thw {base_enc['image_grid_thw'][0].tolist()}")
    print(f"  A vs stored run-1: max abs diff patch_mean {np.abs(base['patch_mean'] - stored['patch_mean']).max():.3g}, "
          f"patch_max {np.abs(base['patch_max'] - stored['patch_max']).max():.3g}")
    for name, (kinds, enc) in results.items():
        same_input = torch.equal(enc["pixel_values"], base_enc["pixel_values"]) and torch.equal(
            enc["image_grid_thw"], base_enc["image_grid_thw"])
        print(f"  {name}: pixel_values equal to A {same_input}; vs A max abs diff "
              f"patch_mean {np.abs(kinds['patch_mean'] - base['patch_mean']).max():.3g}, "
              f"patch_max {np.abs(kinds['patch_max'] - base['patch_max']).max():.3g}; vs vit-1 max abs diff "
              f"patch_mean {np.abs(kinds['patch_mean'] - vit1['patch_mean']).max():.3g}, "
              f"patch_max {np.abs(kinds['patch_max'] - vit1['patch_max']).max():.3g}, "
              f"min cosine patch_max {cosine(kinds['patch_max'], vit1['patch_max']):.6f}")
print(f"\npeak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
