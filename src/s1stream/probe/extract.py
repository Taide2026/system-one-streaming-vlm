"""GPU stage of probing: `s1stream probe-extract` (finetune-layer-probing decisions 3, 5, 6, 8, 14).

Writes $S1_FEATURES/<config name>/<condition>/<run name>/:
  config.yaml          the config as read
  <set>/last.npy       float16 (n_windows, n_states, hidden): every state at the last prompt position
  <set>/visual.npy     float16 (n_windows, n_states, hidden): every state averaged over the image tokens
  <set>/meta.jsonl     one line per window: {i, video, category, window, start_s, end_s, label, holmes, p_yes, seconds}
  run.json             written last: {name, condition, started, finished, free_memory_gb, peak_gpu_gb, load_s,
                       windows: {<set>: n}, limit_windows, versions, git_commit, git_dirty, assets: {name: {revision,
                       sha256}}, visual_weights_equal (A and B: are all visual.* tensors of the runtime and the base
                       weights equal; C: None)}
Sets: train_anomaly, train_normal, test. While a set runs, its windows go to <set>/parts/part-<k>/ (part_size windows
each, with a file "done" when complete); after the last part, the parts are merged into the three files above and
<set>/parts is removed. A rerun into the same folder skips finished sets and finished parts.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from s1stream.config import WindowSpec
from s1stream.probe.model import WindowResult
from s1stream.types import Question, VideoItem, Window

SETS = ("train_anomaly", "train_normal", "test")
EXPECTED_PEAK_GB = 19.0
ATTENTION_MODES = ("causal", "noncausal")


@dataclass(frozen=True)
class ConditionSpec:
    """weights: the asset whose tensors replace the runtime backbone (A, B) or None (C); weights_path: its folder."""
    name: str
    weights: str | None
    weights_path: Path | None
    attention: str


@dataclass(frozen=True)
class ProbeConfig:
    name: str
    train: dict[str, Any]     # dataset spec, load_probe_config adds "path"
    test: dict[str, Any]      # dataset spec, load_probe_config adds "path"
    window: WindowSpec
    question: Question
    input_form: dict[str, Any]
    runtime: str              # asset name of the d3 runtime (d3-flash)
    runtime_path: Path
    memory_cap_gb: float
    conditions: dict[str, ConditionSpec]
    raw: dict[str, Any]       # the YAML as read


@dataclass(frozen=True)
class WindowRow:
    """One window to extract. label: test = annotation overlap, train = video label (1 anomaly, 0 normal).
    holmes: train anomaly = Holmes event overlap (the item's events), train normal = 0, test = None."""
    set: str
    item: VideoItem
    window: Window
    label: int
    holmes: int | None


def load_probe_config(path: Path, resolver: Callable[[str], Path] | None = None) -> ProbeConfig:
    """Read a probe config (example: configs/probe_ucf_short_v1_f8.yaml).

    Sections: name, train, test, window, question, input_form, runtime, memory_cap_gb, conditions.
    - Raise s1stream.config.ConfigError naming the section when one is missing.
    - question: as s1stream.config.load_config (question.prompt names s1stream.prompt.PROMPTS; config keys win;
      positive must be one option, a str).
    - train.path, test.path = resolver(<spec>["asset"]); runtime_path = resolver(runtime); each condition's
      weights_path = resolver(weights) or None. resolver defaults to s1stream.assets.resolve; an AssetError becomes a
      ConfigError naming the asset.
    - conditions: {<name>: {weights: <asset> | null, attention: causal | noncausal}}. Raise ConfigError naming the
      condition for another attention value.
    """
    raise NotImplementedError


def plan_sets(train_items: list[VideoItem], test_items: list[VideoItem], window: WindowSpec) -> dict[str, list[WindowRow]]:
    """{set: rows} for SETS, windows from s1stream.windows.plan_windows, in item order then window order.

    train items with category "Normal" go to train_normal (label 0, holmes 0); other train items go to train_anomaly
    (label 1, holmes = window.label, which plan_windows sets from the item's Holmes events); test items go to test
    (label = window.label, holmes None).
    """
    raise NotImplementedError


def limit_sets(sets: dict[str, list[WindowRow]], limit_windows: int | None) -> dict[str, list[WindowRow]]:
    """The first limit_windows rows of each set; every row when limit_windows is None."""
    raise NotImplementedError


def extract(rows: list[WindowRow], run_window: Callable[[WindowRow], WindowResult], out: Path,
            part_size: int = 500, log: Callable[[str], None] = print) -> None:
    """Extract one set into out (see the module docstring for the files and the parts).

    - Finished set (out/meta.jsonl and both .npy files exist, no out/parts): return without calling run_window.
    - A finished part (its "done" file exists) is not run again. Before it is skipped, its meta rows must match the
      plan (video, window, start_s, end_s, in order); raise RuntimeError naming the part when they differ.
    - An unfinished part is run from its first window again.
    - meta "i" is the row index in rows; "seconds" is the wall time of run_window.
    - Log one line every 200 windows.
    """
    raise NotImplementedError


def write_run_json(out: Path, config: ProbeConfig, condition: str, extra: dict[str, Any]) -> None:
    """Write out/run.json: name, condition, git_commit and git_dirty (s1stream.runner.git_commit of this repo),
    versions (python, torch, transformers), assets: {name: {revision, sha256}} from s1stream.assets.load_lock for
    the train and test dataset assets, the train holmes_asset, the runtime and the condition's weights; then every
    key of extra (started, finished, free_memory_gb, peak_gpu_gb, load_s, windows, limit_windows,
    visual_weights_equal)."""
    raise NotImplementedError


def features_dir(config: ProbeConfig, condition: str, run_name: str) -> Path:
    """$S1_FEATURES/<config name>/<condition>/<run name>; S1_FEATURES defaults to ./features."""
    raise NotImplementedError


def run_extract(config_path: Path, condition: str, run_name: str, limit_windows: int | None = None) -> Path:
    """The probe-extract command. Return the output folder.

    1. load_probe_config; raise ConfigError for a condition that is not in the config.
    2. out = features_dir(...). Raise FileExistsError when out/run.json exists (a finished run is never overwritten).
       Write out/config.yaml.
    3. Raise RuntimeError when s1stream.runner.free_memory_gb() < EXPECTED_PEAK_GB + 10.
    4. Build the datasets (registry DATASETS), plan_sets, limit_sets.
    5. On cuda: torch.cuda.set_per_process_memory_fraction(memory_cap_gb / total memory). load_condition(...)
       (s1stream.probe.model); load_s. Reset peak memory statistics.
    6. For each set: extract(rows, run_window, out / set) with run_window building the images with the input form
       (registry INPUT_FORMS) and calling ProbeModel.run_window.
    7. write_run_json.
    """
    raise NotImplementedError
