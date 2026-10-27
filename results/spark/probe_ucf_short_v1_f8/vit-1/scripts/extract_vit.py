"""Extract vision-tower (ViT) features for the probe windows (2026-10-26 spike).

The d3-flash and Qwen3.5-9B vision towers are the same weights (run-1 extract: visual_weights_equal = true for A, B),
so one extraction covers A, B and C. Same config, windows, frames and processor as probe_ucf_short_v1_f8 run-1.

Per window, only the vision path runs: processor.image_processor on the 8 frames, then
backbone.get_image_features (the call the runtime makes before the LM). Taps:
  ViT states 0..27: 0 = patch embedding + position embedding (input of block 0), s = output of block s - 1.
                    1152 dims, pooled over all patch tokens of the 8 frames.
  merger:           the merger output (4096 dims, the tokens the LM receives), pooled over its tokens.
Two poolings per state: mean and max over tokens.

Writes probe-fit feature folders (probe-fit reads <set>/last.npy and <set>/visual.npy):
  $S1_FEATURES/probe_ucf_short_v1_f8/V/<run>/<set>/{visual,last}.npy  ViT, (n, 28, 1152) float32
  $S1_FEATURES/probe_ucf_short_v1_f8/M/<run>/<set>/{visual,last}.npy  merger, (n, 1, 4096) float32 (float16 would overflow: ViT max-pool values reach about 1.3e4)
  visual.npy = token mean, last.npy = token MAX (not a last token: the ViT has none). meta.jsonl as probe-extract.
  run.json in both folders.

Usage (worktree root):
  uv run python results/spark/probe_ucf_short_v1_f8/vit-1/scripts/extract_vit.py --run vit-1 [--limit-windows N] [--check N]
--check N: before the extraction, run N test windows through the full runtime forward too and compare its merger
output with the vision-only path (they must be equal); then exit.
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


class VisionTap:
    """Hooks on visual.blocks[0] (pre), every block and the merger; pools each output over tokens at once."""

    def __init__(self, visual: torch.nn.Module) -> None:
        self.vit_mean: list[torch.Tensor] = []
        self.vit_max: list[torch.Tensor] = []
        self.merger: list[torch.Tensor] = []
        blocks = list(visual.blocks)
        self.n_states = len(blocks) + 1
        self.hooks = [blocks[0].register_forward_pre_hook(self._pre)]
        self.hooks += [block.register_forward_hook(self._block) for block in blocks]
        self.hooks.append(visual.merger.register_forward_hook(self._merger))

    @staticmethod
    def _pool(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden.detach().float()
        return hidden.mean(dim=0).cpu(), hidden.max(dim=0).values.cpu()

    def _pre(self, module, args, kwargs=None) -> None:
        mean, top = self._pool(args[0])
        self.vit_mean.append(mean)
        self.vit_max.append(top)

    def _block(self, module, args, output) -> None:
        mean, top = self._pool(output[0] if isinstance(output, tuple) else output)
        self.vit_mean.append(mean)
        self.vit_max.append(top)

    def _merger(self, module, args, output) -> None:
        self.merger.append(output.detach())

    def reset(self) -> None:
        self.vit_mean.clear()
        self.vit_max.clear()
        self.merger.clear()

    def collect(self) -> dict[str, np.ndarray]:
        if len(self.vit_mean) != self.n_states or len(self.merger) != 1:
            raise RuntimeError(f"expected {self.n_states} ViT states and 1 merger output, "
                               f"got {len(self.vit_mean)} and {len(self.merger)}")
        merger_mean, merger_max = self._pool(self.merger[0])
        return {"vit_mean": torch.stack(self.vit_mean).numpy(), "vit_max": torch.stack(self.vit_max).numpy(),
                "merger_mean": merger_mean[None].numpy(), "merger_max": merger_max[None].numpy()}


def vision_only(model, images, tap: VisionTap) -> dict[str, np.ndarray]:
    runtime = model.runtime
    sdpa_backends = sys.modules[type(runtime).__module__].sdpa_backends  # the runtime's own attention setting
    with torch.inference_mode(), sdpa_backends(runtime.device):
        tap.reset()
        encoded = runtime.processor.image_processor(images=images, return_tensors="pt")
        runtime.backbone.get_image_features(encoded["pixel_values"].to(runtime.device),
                                            encoded["image_grid_thw"].to(runtime.device), return_dict=True)
        return tap.collect()


def check(model, config, rows, input_form, tap: VisionTap, n: int) -> None:
    for row in rows[:n]:
        images = input_form.build(row.item, row.window, config.question).images
        alone = vision_only(model, images, tap)
        tap.reset()
        model.run_window(images, config.question)
        full = tap.collect()
        for key in alone:
            diff = float(np.abs(alone[key] - full[key]).max())
            scale = float(np.abs(full[key]).max())
            print(f"check {row.item.name} w{row.window.index} {key}: max abs diff {diff:.3g} (max abs {scale:.3g})")
            if diff > 1e-3 * scale:
                raise SystemExit(f"vision-only path differs from the full forward ({key})")
    print(f"check passed on {n} windows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--limit-windows", type=int)
    parser.add_argument("--check", type=int, default=0)
    args = parser.parse_args()

    config = load_probe_config(CONFIG)
    root = Path(os.environ["S1_FEATURES"]) / config.name
    outs = {"V": root / "V" / args.run, "M": root / "M" / args.run}
    if not args.check:
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
    tap = VisionTap(model.runtime.backbone.visual)
    torch.cuda.reset_peak_memory_stats()
    if args.check:
        check(model, config, sets["test"], input_form, tap, args.check)
        print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        return

    seconds_total = {}
    for name, rows in sets.items():
        arrays: dict[str, list[np.ndarray]] = {"vit_mean": [], "vit_max": [], "merger_mean": [], "merger_max": []}
        meta = []
        set_start = time.perf_counter()
        for i, row in enumerate(rows):
            begun = time.perf_counter()
            images = input_form.build(row.item, row.window, config.question).images
            result = vision_only(model, images, tap)
            for key, value in result.items():
                arrays[key].append(value.astype(np.float32))
            meta.append({"i": i, "video": row.item.name, "category": row.item.category, "window": row.window.index,
                         "start_s": row.window.start_s, "end_s": row.window.end_s, "label": row.label,
                         "holmes": row.holmes, "p_yes": math.nan, "seconds": time.perf_counter() - begun})
            if (i + 1) % 500 == 0:
                print(f"{name}: {i + 1}/{len(rows)} windows, {time.perf_counter() - set_start:.0f} s", flush=True)
        seconds_total[name] = time.perf_counter() - set_start
        meta_text = "".join(json.dumps(row) + "\n" for row in meta)
        for folder, prefix in (("V", "vit"), ("M", "merger")):
            out = outs[folder] / name
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "visual.npy", np.stack(arrays[f"{prefix}_mean"]))
            np.save(out / "last.npy", np.stack(arrays[f"{prefix}_max"]))
            (out / "meta.jsonl").write_text(meta_text)
        print(f"{name}: done, {seconds_total[name]:.0f} s", flush=True)

    commit, dirty = git_commit(Path.cwd())
    record = {"name": config.name, "source": "vision tower of d3-flash (= Qwen3.5-9B)", "git_commit": commit,
              "git_dirty": dirty, "started": started, "finished": datetime.now(timezone.utc).isoformat(),
              "free_memory_gb": memory, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30, "load_s": load_s,
              "seconds": seconds_total, "windows": {name: len(rows) for name, rows in sets.items()},
              "limit_windows": args.limit_windows, "kinds": {"visual": "token mean", "last": "token max"}}
    for folder, out in outs.items():
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.json").write_text(json.dumps({**record, "features": folder}, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
