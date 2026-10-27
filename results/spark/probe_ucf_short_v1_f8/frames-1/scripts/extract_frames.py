"""Per-frame vision features for the probe windows (2026-10-27 derived experiment of vit-1).

The vision tower encodes each of the 8 frames as its own image (image_grid_thw: 8 rows, t = 1), so the ViT sees no
motion. vit-1 averaged the tokens of all 8 frames. Here the tokens are averaged per frame, which keeps the order:
  ViT block 27 output (the last block): (8, 1152) per window
  merger output (the tokens the LM receives): (8, 4096) per window
Same windows, frames and vision-only path as vit-1/scripts/extract_vit.py (checked equal to the full forward).

Writes probe-fit feature folders with one state each:
  $S1_FEATURES/probe_ucf_short_v1_f8/FV/<run>/  ViT block 27
  $S1_FEATURES/probe_ucf_short_v1_f8/FM/<run>/  merger
  last.npy   = the 8 frame means concatenated in time order, (n, 1, 8 d) float32   ("concat")
  visual.npy = [mean over frames, std over frames], (n, 1, 2 d) float32            ("meanstd")
  frames.npy = the raw (n, 8, d) per-frame means, for later read-outs.

Usage (worktree root):
  uv run python results/spark/probe_ucf_short_v1_f8/frames-1/scripts/extract_frames.py --run frames-1 [--limit-windows N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import s1stream.plugins  # noqa: F401  Register the plugins.
from s1stream.probe.extract import limit_sets, load_probe_config, plan_sets
from s1stream.probe.model import load_condition
from s1stream.registry import DATASETS, INPUT_FORMS
from s1stream.runner import free_memory_gb, git_commit

CONFIG = Path("configs/probe_ucf_short_v1_f8.yaml")
EXPECTED_PEAK_GB = 19.0
MEMORY_CAP_GB = 40.0
FRAMES = 8


def per_frame(hidden: torch.Tensor, sizes: list[int]) -> np.ndarray:
    """(tokens, d) -> (frames, d): mean over each frame's tokens (the tokens are in frame order)."""
    return torch.stack([part.float().mean(dim=0) for part in hidden.detach().split(sizes)]).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--limit-windows", type=int)
    args = parser.parse_args()

    config = load_probe_config(CONFIG)
    root = Path(os.environ["S1_FEATURES"]) / config.name
    outs = {"FV": root / "FV" / args.run, "FM": root / "FM" / args.run}
    for out in outs.values():
        if (out / "run.json").exists():
            raise FileExistsError(out / "run.json")
    memory = free_memory_gb()
    if memory is not None and memory < EXPECTED_PEAK_GB + 10:
        raise SystemExit(f"Insufficient available memory: {memory:.1f} GiB")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GB / total)
    started = datetime.now(timezone.utc).isoformat()
    sets = limit_sets(plan_sets(DATASETS.build(config.train).items(), DATASETS.build(config.test).items(),
                                config.window), args.limit_windows)
    input_form = INPUT_FORMS.build(config.input_form)
    load_start = time.perf_counter()
    model = load_condition("C", config.runtime_path, None, "noncausal", device="cuda")
    load_s = time.perf_counter() - load_start
    runtime = model.runtime
    visual = runtime.backbone.visual
    merge = visual.config.spatial_merge_size ** 2
    sdpa_backends = sys.modules[type(runtime).__module__].sdpa_backends
    captured: dict[str, torch.Tensor] = {}
    hooks = [visual.blocks[-1].register_forward_hook(
                 lambda m, a, o: captured.__setitem__("vit", o[0] if isinstance(o, tuple) else o)),
             visual.merger.register_forward_hook(lambda m, a, o: captured.__setitem__("merger", o))]
    torch.cuda.reset_peak_memory_stats()

    seconds_total = {}
    for name, rows in sets.items():
        frames = {"FV": [], "FM": []}
        meta = []
        set_start = time.perf_counter()
        for i, row in enumerate(rows):
            begun = time.perf_counter()
            images = input_form.build(row.item, row.window, config.question).images
            with torch.inference_mode(), sdpa_backends(runtime.device):
                captured.clear()
                encoded = runtime.processor.image_processor(images=images, return_tensors="pt")
                grid = encoded["image_grid_thw"]
                if grid.shape[0] != FRAMES or bool((grid[:, 0] != 1).any()):
                    raise RuntimeError(f"expected {FRAMES} images with t = 1, got grid {grid.tolist()}")
                runtime.backbone.get_image_features(encoded["pixel_values"].to(runtime.device),
                                                    grid.to(runtime.device), return_dict=True)
                sizes = grid.prod(dim=1).tolist()
                frames["FV"].append(per_frame(captured["vit"], sizes))
                frames["FM"].append(per_frame(captured["merger"], [s // merge for s in sizes]))
            meta.append({"i": i, "video": row.item.name, "category": row.item.category, "window": row.window.index,
                         "start_s": row.window.start_s, "end_s": row.window.end_s, "label": row.label,
                         "holmes": row.holmes, "p_yes": math.nan, "seconds": time.perf_counter() - begun})
            if (i + 1) % 500 == 0:
                print(f"{name}: {i + 1}/{len(rows)} windows, {time.perf_counter() - set_start:.0f} s", flush=True)
        seconds_total[name] = time.perf_counter() - set_start
        meta_text = "".join(json.dumps(row) + "\n" for row in meta)
        for folder, out in outs.items():
            data = np.stack(frames[folder]).astype(np.float32)  # (n, 8, d)
            target = out / name
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / "frames.npy", data)
            np.save(target / "last.npy", data.reshape(len(data), 1, -1))
            np.save(target / "visual.npy", np.concatenate([data.mean(axis=1), data.std(axis=1)], axis=1)[:, None])
            (target / "meta.jsonl").write_text(meta_text)
        print(f"{name}: done, {seconds_total[name]:.0f} s", flush=True)
    for hook in hooks:
        hook.remove()

    commit, dirty = git_commit(Path.cwd())
    record = {"name": config.name, "source": "vision tower of d3-flash (= Qwen3.5-9B), per frame",
              "git_commit": commit, "git_dirty": dirty, "started": started,
              "finished": datetime.now(timezone.utc).isoformat(), "free_memory_gb": memory,
              "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30, "load_s": load_s, "seconds": seconds_total,
              "windows": {name: len(rows) for name, rows in sets.items()}, "limit_windows": args.limit_windows,
              "kinds": {"last": "concat of 8 frame means", "visual": "mean and std over frames"}}
    for folder, out in outs.items():
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.json").write_text(json.dumps({**record, "features": folder}, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
