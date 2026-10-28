"""GPU stage of encoder probing: `s1stream encoder-extract` (encoder-probing decisions 2, 3, 5, 10).

Writes $S1_FEATURES/<config name>/<encoder>/<run name>/ and, when the adapter has head kinds,
$S1_FEATURES/<config name>/<encoder>-head/<run name>/. Each folder:
  config.yaml        the config as read
  <set>/<kind>.npy   float32 (n_windows, n_states, dims); the head folder has n_states = 1
  <set>/meta.jsonl   one line per window: {i, video, category, window, start_s, end_s, label, holmes, p_yes, seconds}
                     (p_yes is null: an encoder has no answer); the same file in both folders
  run.json           written last: {name, encoder, folder ("main" | "head"), kinds: [...], n_states, dims,
                     git_commit, git_dirty, versions, assets: {name: {revision, sha256}}, started, finished,
                     free_memory_gb, peak_gpu_gb, load_s, seconds: {<set>: s}, windows: {<set>: n}, limit_windows}
Sets: train_anomaly, train_normal, test (probe.extract.SETS).
"""
from __future__ import annotations

import copy
import json
import os
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from s1stream.assets import AssetError, LOCK_FILE, load_lock, resolve
from s1stream.config import ConfigError, WindowSpec
from s1stream.encoder.adapters import ENCODERS, load_encoder
from s1stream.probe.extract import SETS, WindowRow, limit_sets, plan_sets

EXPECTED_PEAK_GB = 10.0


@dataclass(frozen=True)
class EncoderConfig:
    name: str
    train: dict[str, Any]          # dataset spec, load_encoder_config adds "path"
    test: dict[str, Any]           # dataset spec, load_encoder_config adds "path"
    window: WindowSpec
    input_form: dict[str, Any]
    memory_cap_gb: float
    encoders: dict[str, str]       # encoder name -> asset name
    encoder_paths: dict[str, Path] # encoder name -> resolved asset folder
    raw: dict[str, Any]            # the YAML as read


def load_encoder_config(path: Path, resolver: Callable[[str], Path] | None = None) -> EncoderConfig:
    """Read an encoder config (example: configs/encoder_probe_ucf_short_v1_f8.yaml).
    Sections: name, train, test, window, input_form, memory_cap_gb, encoders. Raise ConfigError naming a missing
    section, and naming an encoder that is not in adapters.ENCODERS. train.path, test.path and encoder_paths come from
    resolver (default s1stream.assets.resolve); an AssetError becomes a ConfigError naming the asset."""
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read config: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("Config must be a mapping")
    for section in ("name", "train", "test", "window", "input_form", "memory_cap_gb", "encoders"):
        if section not in data:
            raise ConfigError(f"Missing section: {section}")
    for name in data["encoders"]:
        if name not in ENCODERS:
            raise ConfigError(f"Unknown encoder {name!r}")
    raw = copy.deepcopy(data)
    if resolver is None:
        resolver = resolve

    def asset_path(name: str) -> Path:
        try:
            return resolver(name)
        except AssetError as exc:
            raise ConfigError(f"Could not resolve asset {name!r}: {exc}") from exc

    for section in ("train", "test"):
        data[section]["path"] = asset_path(data[section]["asset"])
    paths = {name: asset_path(asset) for name, asset in data["encoders"].items()}
    return EncoderConfig(data["name"], data["train"], data["test"], WindowSpec(**data["window"]),
                         data["input_form"], data["memory_cap_gb"], data["encoders"], paths, raw)


def features_dir(config: EncoderConfig, folder: str, run_name: str) -> Path:
    """$S1_FEATURES/<config name>/<folder>/<run name>; folder is the encoder name or "<encoder>-head".
    S1_FEATURES defaults to ./features."""
    return Path(os.environ.get("S1_FEATURES") or Path.cwd() / "features") / config.name / folder / run_name


def run_encoder_extract(config_path: Path, encoder: str, run_name: str, limit_windows: int | None = None,
                        loader: Callable[[str, Path, str], Any] | None = None,
                        plan: Callable[[EncoderConfig], dict[str, list[WindowRow]]] | None = None,
                        frames: Callable[[WindowRow], list[Image.Image]] | None = None,
                        resolver: Callable[[str], Path] | None = None) -> Path:
    """The encoder-extract command. Return the main folder.

    1. load_encoder_config(config_path, resolver). Raise ConfigError when encoder is not in config.encoders.
    2. Raise FileExistsError when run.json exists in the main folder or in the head folder.
    3. Raise RuntimeError when s1stream.runner.free_memory_gb() < EXPECTED_PEAK_GB + 10 (None: no check).
       Look free_memory_gb up on s1stream.runner at call time (from s1stream import runner; runner.free_memory_gb()).
    4. plan(config) (default: registry DATASETS for train and test, probe.extract.plan_sets), then limit_sets.
    5. On cuda: torch.cuda.set_per_process_memory_fraction(memory_cap_gb / total memory).
       adapter = loader(encoder, config.encoder_paths[encoder], device) (default adapters.load_encoder); load_s.
    6. For each set and window: frames(row) (default: registry INPUT_FORMS from config.input_form, .build(item,
       window, None).images), adapter.encode(frames); write the arrays and meta of the set into both folders.
    7. Write config.yaml and run.json into each folder (head folder only when adapter.head_kinds is not empty).
    """
    from s1stream import runner
    from s1stream.registry import DATASETS, INPUT_FORMS

    config = load_encoder_config(config_path, resolver)
    if encoder not in config.encoders:
        raise ConfigError(f"Unknown configured encoder {encoder!r}")
    out = features_dir(config, encoder, run_name)
    head_out = features_dir(config, f"{encoder}-head", run_name)
    for folder in (out, head_out):
        if (folder / "run.json").exists():
            raise FileExistsError(folder / "run.json")
    started = datetime.now(timezone.utc).isoformat()
    memory = runner.free_memory_gb()
    if memory is not None and memory < EXPECTED_PEAK_GB + 10:
        raise RuntimeError(f"Insufficient available memory: {memory:.1f} GiB")
    if plan is None:
        def default_plan(cfg: EncoderConfig) -> dict[str, list[WindowRow]]:
            return plan_sets(DATASETS.build(cfg.train).items(), DATASETS.build(cfg.test).items(), cfg.window)

        plan = default_plan
    sets = limit_sets(plan(config), limit_windows)
    if frames is None:
        input_form = INPUT_FORMS.build(config.input_form)
        def default_frames(row: WindowRow) -> list[Image.Image]:
            return input_form.build(row.item, row.window, None).images

        frames = default_frames
    if loader is None:
        loader = load_encoder
    cuda = torch.cuda.is_available()
    if cuda:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(config.memory_cap_gb / total)
    load_start = time.perf_counter()
    adapter = loader(encoder, config.encoder_paths[encoder], "cuda" if cuda else "cpu")
    load_s = time.perf_counter() - load_start
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    folders = [(out, "main", adapter.kinds)]
    if adapter.head_kinds:
        folders.append((head_out, "head", adapter.head_kinds))
    for folder, _, _ in folders:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.yaml").write_text(yaml.safe_dump(config.raw, sort_keys=False))
    seconds_by_set = {}
    dims = {"main": 0, "head": 0}
    for name in SETS:
        begun = time.perf_counter()
        rows = sets[name]
        arrays = {"main": {kind: [] for kind in adapter.kinds},
                  "head": {kind: [] for kind in adapter.head_kinds}}
        meta = []
        for i, row in enumerate(rows):
            if cuda:
                torch.cuda.synchronize()
            window_start = time.perf_counter()
            with torch.inference_mode():
                result = adapter.encode(frames(row))
            if cuda:
                torch.cuda.synchronize()
            meta.append(window_meta(i, row, time.perf_counter() - window_start))
            for group, values in (("main", result.kinds), ("head", result.head)):
                for kind, tensor in values.items():
                    array = tensor.numpy()
                    arrays[group][kind].append(array)
                    dims[group] = array.shape[-1]
            if (i + 1) % 500 == 0:
                print(f"{name}: {i + 1} windows, {time.perf_counter() - begun:.1f} seconds", flush=True)
        for folder, group, kinds in folders:
            set_dir = folder / name
            set_dir.mkdir(exist_ok=True)
            for kind in kinds:
                values = arrays[group][kind]
                states = adapter.n_states if group == "main" else 1
                stacked = np.stack(values) if values else np.empty((0, states, dims[group]), dtype=np.float32)
                np.save(set_dir / f"{kind}.npy", stacked)
            (set_dir / "meta.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in meta))
        seconds_by_set[name] = time.perf_counter() - begun

    names = {config.raw["train"]["asset"], config.raw["test"]["asset"], config.encoders[encoder]}
    if "holmes_asset" in config.raw["train"]:
        names.add(config.raw["train"]["holmes_asset"])
    lock = load_lock(LOCK_FILE)
    commit, dirty = runner.git_commit(Path(__file__).resolve().parents[3])
    common = {"name": config.name, "encoder": encoder, "git_commit": commit, "git_dirty": dirty,
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "transformers": version("transformers")},
              "assets": {name: {"revision": asset.revision, "sha256": asset.sha256}
                         for name in sorted(names) if (asset := lock.get(name)) is not None},
              "started": started, "finished": datetime.now(timezone.utc).isoformat(),
              "free_memory_gb": memory, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
              "load_s": load_s, "seconds": seconds_by_set,
              "windows": {name: len(sets[name]) for name in SETS}, "limit_windows": limit_windows}
    for folder, group, kinds in folders:
        data = {**common, "folder": group, "kinds": list(kinds),
                "n_states": adapter.n_states if group == "main" else 1, "dims": dims[group]}
        (folder / "run.json").write_text(json.dumps(data, indent=1))
    return out


def window_meta(i: int, row: WindowRow, seconds: float) -> dict:
    """One meta.jsonl row (see the module docstring), p_yes None."""
    return {"i": i, "video": row.item.name, "category": row.item.category,
            "window": row.window.index, "start_s": row.window.start_s, "end_s": row.window.end_s,
            "label": row.label, "holmes": row.holmes, "p_yes": None, "seconds": seconds}
