"""Prototype: image vs video preprocessing of one window (no model load). Prints grids, shapes, processor settings."""
import inspect
from pathlib import Path

import numpy as np
import torch

import s1stream.plugins  # noqa: F401
from s1stream.probe.extract import load_probe_config, plan_sets
from s1stream.registry import DATASETS, INPUT_FORMS
from transformers import AutoProcessor

config = load_probe_config(Path("configs/probe_ucf_short_v1_f8.yaml"))
sets = plan_sets([], DATASETS.build(config.test).items()[:1], config.window)
row = sets["test"][0]
images = INPUT_FORMS.build(config.input_form).build(row.item, row.window, config.question).images
print("frames", len(images), images[0].size, row.window.frame_indices)
proc = AutoProcessor.from_pretrained(config.runtime_path, local_files_only=True)
vp = proc.video_processor
print("video processor", type(vp).__name__, {k: getattr(vp, k, None) for k in ("do_sample_frames", "fps", "min_frames", "max_frames", "temporal_patch_size", "size", "patch_size", "merge_size")})
img = proc.image_processor(images=images, return_tensors="pt")
print("image grid", img["image_grid_thw"].tolist(), tuple(img["pixel_values"].shape))
vid = vp(videos=[np.stack([np.asarray(im) for im in images])], return_tensors="pt", do_sample_frames=False)
print("video keys", list(vid.keys()))
print("video grid", vid["video_grid_thw"].tolist(), tuple(vid["pixel_values_videos"].shape))
# pixel check: video pixels for temporal patch 0 should hold frames 0 and 1; image pixels hold frame 0 twice
c, tp, p = 3, 2, 16
iv = img["pixel_values"].view(8, -1, c, tp, p, p)
vv = vid["pixel_values_videos"].view(4, -1, c, tp, p, p) if vid["video_grid_thw"][0, 0] == 4 else None
print("image frame0 t0==t1:", bool(torch.equal(iv[0, :, :, 0], iv[0, :, :, 1])))
if vv is not None:
    print("video slice0 t0 vs image frame0:", float((vv[0, :, :, 0] - iv[0, :, :, 0]).abs().max()),
          "| t1 vs image frame1:", float((vv[0, :, :, 1] - iv[1, :, :, 0]).abs().max()),
          "| slice1 t0 vs image frame2:", float((vv[1, :, :, 0] - iv[2, :, :, 0]).abs().max()))
