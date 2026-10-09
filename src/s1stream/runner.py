"""GPU stage: `s1stream run <config>` (proposal decisions 7, 13, 14).

Writes into a new run folder:
  config.yaml    the config as read (ExperimentConfig.raw)
  manifest.json  the selected videos: [{name, category, fps, n_frames, duration_s, events, path}]
  scores.jsonl   one line per window, written and flushed as it is scored:
                 {video, window, start_s, end_s, label, score, probabilities, seconds}
                 score = probabilities[question.positive]; seconds = wall time of input form + backbone + head.
  run.json       {name, started, finished, free_memory_gb, peak_gpu_gb, load_s, windows, limit_windows,
                  versions: {python, torch, transformers}, git_commit, git_dirty, assets: {name: {revision, sha256}}}
"""
from __future__ import annotations

from pathlib import Path

from s1stream.config import ExperimentConfig


def free_memory_gb() -> float | None:
    """MemAvailable from /proc/meminfo in GiB, or None when the file cannot be read."""
    raise NotImplementedError


def git_commit(repo: Path) -> tuple[str | None, bool]:
    """(HEAD commit, True when the work tree has uncommitted changes); (None, False) when git fails."""
    raise NotImplementedError


def run(config: ExperimentConfig, out_dir: Path, limit_windows: int | None = None) -> Path:
    """Run the GPU stage and return out_dir.

    1. Raise FileExistsError when out_dir exists. Create it (parents too).
    2. Write config.yaml. Build the dataset (registry DATASETS) and write manifest.json.
    3. Build the input form, then the backbone and the answer head; load_s = wall time of the backbone and head builds.
       On cuda, reset the peak memory statistics after loading.
    4. For each video, for each window of plan_windows(item, window.length_s, window.stride_s, window.frames):
       build the input, run the backbone, get probabilities, append one line to scores.jsonl, flush.
       Stop after limit_windows lines when it is set. Print one progress line per video.
    5. Write run.json. peak_gpu_gb is torch.cuda.max_memory_allocated() / 2**30 on cuda, else None.
       assets lists every spec of the config with an "asset" key, with revision and sha256 from s1stream.assets.load_lock.
    """
    raise NotImplementedError
