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

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from s1stream.config import ConfigError, WindowSpec
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
    raise NotImplementedError


def features_dir(config: EncoderConfig, folder: str, run_name: str) -> Path:
    """$S1_FEATURES/<config name>/<folder>/<run name>; folder is the encoder name or "<encoder>-head".
    S1_FEATURES defaults to ./features."""
    raise NotImplementedError


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
    raise NotImplementedError


def window_meta(i: int, row: WindowRow, seconds: float) -> dict:
    """One meta.jsonl row (see the module docstring), p_yes None."""
    raise NotImplementedError
