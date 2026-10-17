"""GPU stage: `s1stream run <config>` (proposal decisions 7, 13, 14).

Writes into a new run folder:
  config.yaml    the config as read (ExperimentConfig.raw)
  manifest.json  the selected videos: [{name, category, fps, n_frames, duration_s, events, path}]
                 path is relative to the dataset's asset folder, so no machine path leaves the machine
  scores.jsonl   one line per window, written and flushed as it is scored:
                 {video, window, start_s, end_s, label, score, probabilities, seconds}
                 score = probabilities[question.positive]; seconds = wall time of input form + backbone + head.
                 A backbone that decodes adds text, its answer.
  run.json       {name, started, finished, free_memory_gb, peak_gpu_gb, load_s, windows, limit_windows,
                  versions: {python, torch, transformers}, git_commit, git_dirty, assets: {name: {revision, sha256}}}
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
import yaml

from s1stream.assets import LOCK_FILE, load_lock
from s1stream.config import ExperimentConfig
from s1stream.registry import ANSWER_HEADS, BACKBONES, DATASETS, INPUT_FORMS
from s1stream.windows import plan_windows


def free_memory_gb() -> float | None:
    """MemAvailable from /proc/meminfo in GiB, or None when the file cannot be read."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 2**30
    except (OSError, ValueError, IndexError):
        return None
    return None


def manifest_path(path: Path, root: str | Path | None) -> str:
    """path relative to root when path lies under root; else path unchanged."""
    if root is None or not Path(path).is_relative_to(root):
        return str(path)
    return str(Path(path).relative_to(root))


def git_commit(repo: Path) -> tuple[str | None, bool]:
    """(HEAD commit, True when the work tree has uncommitted changes); (None, False) when git fails."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
    except OSError:
        return None, False
    if head.returncode or status.returncode:
        return None, False
    return head.stdout.strip(), bool(status.stdout.strip())


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
    out_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    memory = free_memory_gb()
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config.raw, sort_keys=False))
    items = DATASETS.build(config.dataset).items()
    manifest = [{"name": item.name, "category": item.category, "fps": item.fps,
                 "n_frames": item.n_frames, "duration_s": item.duration_s,
                 "events": [list(event) for event in item.events], "path": manifest_path(item.path, config.dataset.get("path"))} for item in items]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    input_form = INPUT_FORMS.build(config.input_form)
    load_start = time.perf_counter()
    backbone = BACKBONES.build(config.backbone)
    head = ANSWER_HEADS.build(config.answer_head)
    load_s = time.perf_counter() - load_start
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    count = 0
    with (out_dir / "scores.jsonl").open("w") as stream:
        for item in items:
            if limit_windows is not None and count >= limit_windows:
                break
            video_count = 0
            for window in plan_windows(item, config.window.length_s, config.window.stride_s,
                                       config.window.frames):
                if limit_windows is not None and count >= limit_windows:
                    break
                if cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    inp = input_form.build(item, window, config.question)
                    output = backbone.forward(inp, config.question)
                    probabilities = head.probabilities(output, config.question)
                if cuda:
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                row = {"video": item.name, "window": window.index, "start_s": window.start_s,
                       "end_s": window.end_s, "label": window.label,
                       "score": probabilities[config.question.positive],
                       "probabilities": probabilities, "seconds": seconds}
                if output.text is not None:
                    row["text"] = output.text
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                count += 1
                video_count += 1
            print(f"{item.name}: {video_count} windows")

    specs = [config.raw[key] for key in ("dataset", "input_form", "backbone", "answer_head")]
    specs.extend(config.raw["decision_rules"])
    asset_names = {spec["asset"] for spec in specs if "asset" in spec}
    assets = {}
    if asset_names:
        lock = load_lock(LOCK_FILE)
        assets = {name: {"revision": lock[name].revision, "sha256": lock[name].sha256}
                  for name in sorted(asset_names)}
    commit, dirty = git_commit(Path(__file__).resolve().parents[2])
    data = {"name": config.name, "started": started, "finished": datetime.now(timezone.utc).isoformat(),
            "free_memory_gb": memory, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
            "load_s": load_s, "windows": count, "limit_windows": limit_windows,
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": version("transformers")},
            "git_commit": commit, "git_dirty": dirty, "assets": assets}
    (out_dir / "run.json").write_text(json.dumps(data, indent=1))
    return out_dir
