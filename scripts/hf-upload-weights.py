#!/usr/bin/env python3
"""Push the trained weights of results/<machine>/ to a Hugging Face model repo (2026-10-28): they are gitignored
(LoRA adapters 627 MB each) and would otherwise live only on pro6000.

  uv run --with huggingface_hub python scripts/hf-upload-weights.py [--repo <user>/<name>] [--public] [--dry-run]

Token: HF_TOKEN or the saved login (~/.cache/huggingface/token). Uploads, per training run under results/*/ft_*/<time>/,
its adapter/ (peft), head.safetensors / readout.safetensors, calibration.json, config.yaml and run.json at the same
relative path, plus a README model card. A run already in the repo (same files) is skipped by the hub's dedup.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

README = """---
license: apache-2.0
tags: [video-anomaly-detection, streaming, lora, peft, qwen3.5, decision-models]
---
# system-one-streaming-vlm: fine-tuned weights

LoRA adapters and read-out heads from the read-out-level fine-tuning of decision models for streaming video anomaly
detection (2 s windows, one forward pass per window, no text decoding). Code, configs, logs and scores:
https://github.com/Taide2026/system-one-streaming-vlm (branch `experiment/kv-across-windows`). This repo holds only the
weights that are too large for git.

## Layout

`results/pro6000/<run name>/<time>/` mirrors the code repo's results folder:

- `adapter/` — peft LoRA (rank 64, every language-model projection of the Qwen3.5 backbone); absent for a linear probe.
- `readout.safetensors` (d3: the 255-code read-out) or `head.safetensors` (Clef-Flash: the joint schema head), float32.
- `calibration.json` — the temperature fitted on held-out training videos; `config.yaml`, `run.json` — the training config.

Run names: `ft_<dataset>_v1_<model>[_variant]`: dataset `ucf` (UCF-Crime train, HIVAU-70k event times) or `xd`
(XD-Violence train); model `d3nano`, `d3flash`, `clef` (Clef-Flash), `qwen` (Qwen3.5-9B control), `qwen2b`; variants
`hard` (hard labels), `probe` (read-out only), `f4`, `p10`, `p30`, `full`, `hardneg` (ablations).

## Use

Point the code repo's eval config at a run folder: `backbone: {name: d3, asset: d3-nano, adapter: results/pro6000/ft_ucf_v1_d3nano_hard/<time>}`.
The adapter is merged at load; inference is unchanged.

Base models: d3-nano / d3-flash (vllm-sr), Clef-Flash (Cloudflare), Qwen3.5-2B / 9B (Apache-2.0). Training data:
UCF-Crime and XD-Violence training videos only; no test video was read.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    repo = args.repo or f"{user}/system-one-streaming-vlm-weights"
    runs = sorted(d for d in args.results.glob("*/ft_*/*") if (d / "adapter").is_dir() or list(d.glob("*.safetensors")))
    total = sum(f.stat().st_size for d in runs for f in d.rglob("*") if f.is_file()) / 1e9
    print(f"{repo}: {len(runs)} runs, {total:.1f} GB")
    for d in runs:
        print(" ", d)
    if args.dry_run:
        return
    api.create_repo(repo, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md", repo_id=repo, repo_type="model")
    for d in runs:
        api.upload_folder(folder_path=str(d), path_in_repo=str(d), repo_id=repo, repo_type="model",
                          allow_patterns=["adapter/*", "*.safetensors", "calibration.json", "config.yaml", "run.json"],
                          commit_message=f"weights: {d}")
        print(f"uploaded {d}", flush=True)
    print(f"done: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
