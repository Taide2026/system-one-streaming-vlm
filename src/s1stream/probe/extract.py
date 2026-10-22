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

import copy
import json
import os
import platform
import shutil
import time
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import version
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import yaml

from s1stream.assets import AssetError, LOCK_FILE, load_lock, resolve
from s1stream.config import ConfigError, WindowSpec
from s1stream.probe.model import WindowResult
from s1stream.types import Question, VideoItem, Window
from s1stream.windows import plan_windows

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
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read config: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("Config must be a mapping")
    for section in ("name", "train", "test", "window", "question", "input_form", "runtime",
                    "memory_cap_gb", "conditions"):
        if section not in data:
            raise ConfigError(f"Missing section: {section}")
    question_data = data["question"]
    if "prompt" in question_data:
        from s1stream.prompt import PROMPTS

        name = question_data["prompt"]
        if name not in PROMPTS:
            raise ConfigError(f"Unknown question.prompt {name!r}: one of {', '.join(PROMPTS)}")
        question_data = data["question"] = {"prompt": name, **copy.deepcopy(PROMPTS[name]),
                                             **{key: value for key, value in question_data.items() if key != "prompt"}}
    positive = question_data["positive"]
    if not isinstance(positive, str) or positive not in question_data["options"]:
        raise ConfigError("question.positive must be a str in question.options")
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
    runtime_path = asset_path(data["runtime"])
    conditions = {}
    for name, spec in data["conditions"].items():
        attention = spec["attention"]
        if attention not in ATTENTION_MODES:
            raise ConfigError(f"condition {name!r}: unknown attention mode {attention!r}")
        weights = spec["weights"]
        conditions[name] = ConditionSpec(name, weights, asset_path(weights) if weights is not None else None,
                                         attention)
    return ProbeConfig(data["name"], data["train"], data["test"], WindowSpec(**data["window"]),
                       Question(question_data["state"], question_data["text"],
                                tuple(question_data["options"]), positive),
                       data["input_form"], data["runtime"], runtime_path, data["memory_cap_gb"],
                       conditions, raw)


def plan_sets(train_items: list[VideoItem], test_items: list[VideoItem], window: WindowSpec) -> dict[str, list[WindowRow]]:
    """{set: rows} for SETS, windows from s1stream.windows.plan_windows, in item order then window order.

    train items with category "Normal" go to train_normal (label 0, holmes 0); other train items go to train_anomaly
    (label 1, holmes = window.label, which plan_windows sets from the item's Holmes events); test items go to test
    (label = window.label, holmes None).
    """
    sets: dict[str, list[WindowRow]] = {name: [] for name in SETS}
    for item in train_items:
        normal = item.category == "Normal"
        name = "train_normal" if normal else "train_anomaly"
        for win in plan_windows(item, window.length_s, window.stride_s, window.frames):
            sets[name].append(WindowRow(name, item, win, 0 if normal else 1, 0 if normal else win.label))
    for item in test_items:
        for win in plan_windows(item, window.length_s, window.stride_s, window.frames):
            sets["test"].append(WindowRow("test", item, win, win.label, None))
    return sets


def limit_sets(sets: dict[str, list[WindowRow]], limit_windows: int | None) -> dict[str, list[WindowRow]]:
    """The first limit_windows rows of each set; every row when limit_windows is None."""
    if limit_windows is None:
        return sets
    return {name: rows[:limit_windows] for name, rows in sets.items()}


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
    if (out / "meta.jsonl").exists() and all((out / f"{kind}.npy").exists() for kind in ("last", "visual")) \
            and not (out / "parts").exists():
        return
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), part_size):
        part = parts / f"part-{start // part_size}"
        batch = rows[start:start + part_size]
        if (part / "done").exists():
            try:
                saved = [json.loads(line) for line in (part / "meta.jsonl").read_text().splitlines()]
                actual = [(row["video"], row["window"], row["start_s"], row["end_s"]) for row in saved]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(f"Finished part {part} has invalid metadata") from exc
            expected = [(row.item.name, row.window.index, row.window.start_s, row.window.end_s) for row in batch]
            if actual != expected:
                raise RuntimeError(f"Finished part {part} differs from the plan")
            continue
        part.mkdir(exist_ok=True)
        last, visual, meta = [], [], []
        for i, row in enumerate(batch, start):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            begun = time.perf_counter()
            result = run_window(row)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds = time.perf_counter() - begun
            last.append(result.last.detach().cpu().numpy().astype(np.float16))
            visual.append(result.visual.detach().cpu().numpy().astype(np.float16))
            meta.append({"i": i, "video": row.item.name, "category": row.item.category,
                         "window": row.window.index, "start_s": row.window.start_s,
                         "end_s": row.window.end_s, "label": row.label, "holmes": row.holmes,
                         "p_yes": result.p_yes, "seconds": seconds})
            if (i + 1) % 200 == 0:
                log(f"{out.name}: {i + 1}/{len(rows)} windows")
        np.save(part / "last.npy", np.stack(last))
        np.save(part / "visual.npy", np.stack(visual))
        (part / "meta.jsonl").write_text("".join(json.dumps(row) + "\n" for row in meta))
        (part / "done").touch()

    part_paths = [parts / f"part-{start // part_size}" for start in range(0, len(rows), part_size)]
    for kind in ("last", "visual"):
        first = np.load(part_paths[0] / f"{kind}.npy", mmap_mode="r") if part_paths else np.empty((0, 0, 0), dtype=np.float16)
        tmp = out / f"{kind}.tmp.npy"
        merged = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                           shape=(len(rows), *first.shape[1:]))
        offset = 0
        for part in part_paths:
            chunk = np.load(part / f"{kind}.npy", mmap_mode="r")
            merged[offset:offset + len(chunk)] = chunk
            offset += len(chunk)
        merged.flush()
        del merged
        tmp.replace(out / f"{kind}.npy")
    tmp_meta = out / "meta.tmp.jsonl"
    with tmp_meta.open("w") as merged_meta:
        for part in part_paths:
            with (part / "meta.jsonl").open() as source:
                shutil.copyfileobj(source, merged_meta)
    tmp_meta.replace(out / "meta.jsonl")
    try:
        shutil.rmtree(parts)
    except OSError as exc:
        raise RuntimeError(f"Could not remove merged parts in {parts}") from exc


def write_run_json(out: Path, config: ProbeConfig, condition: str, extra: dict[str, Any]) -> None:
    """Write out/run.json: name, condition, git_commit and git_dirty (s1stream.runner.git_commit of this repo),
    versions (python, torch, transformers), assets: {name: {revision, sha256}} from s1stream.assets.load_lock for
    the train and test dataset assets, the train holmes_asset, the runtime and the condition's weights; then every
    key of extra (started, finished, free_memory_gb, peak_gpu_gb, load_s, windows, limit_windows,
    visual_weights_equal)."""
    from s1stream.runner import git_commit

    names = {config.raw["train"]["asset"], config.raw["test"]["asset"], config.runtime}
    if "holmes_asset" in config.raw["train"]:
        names.add(config.raw["train"]["holmes_asset"])
    if config.conditions[condition].weights is not None:
        names.add(config.conditions[condition].weights)
    lock = load_lock(LOCK_FILE)
    commit, dirty = git_commit(Path(__file__).resolve().parents[3])
    data = {"name": config.name, "condition": condition, "git_commit": commit, "git_dirty": dirty,
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": version("transformers")},
            "assets": {name: {"revision": lock[name].revision, "sha256": lock[name].sha256}
                       for name in sorted(names)}, **extra}
    (out / "run.json").write_text(json.dumps(data, indent=1))


def features_dir(config: ProbeConfig, condition: str, run_name: str) -> Path:
    """$S1_FEATURES/<config name>/<condition>/<run name>; S1_FEATURES defaults to ./features."""
    return Path(os.environ.get("S1_FEATURES") or Path.cwd() / "features") / config.name / condition / run_name


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
    from s1stream.probe.model import load_condition, rename_base_weights
    from s1stream.registry import DATASETS, INPUT_FORMS
    from s1stream.runner import free_memory_gb

    config = load_probe_config(config_path)
    if condition not in config.conditions:
        raise ConfigError(f"Unknown condition {condition!r}")
    out = features_dir(config, condition, run_name)
    if (out / "run.json").exists():
        raise FileExistsError(out / "run.json")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config.raw, sort_keys=False))
    started = datetime.now(timezone.utc).isoformat()
    memory = free_memory_gb()
    if memory is not None and memory < EXPECTED_PEAK_GB + 10:
        raise RuntimeError(f"Insufficient available memory: {memory:.1f} GiB")
    sets = limit_sets(plan_sets(DATASETS.build(config.train).items(), DATASETS.build(config.test).items(),
                                config.window), limit_windows)
    input_form = INPUT_FORMS.build(config.input_form)
    cuda = torch.cuda.is_available()
    if cuda:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(config.memory_cap_gb / total)
    spec = config.conditions[condition]
    load_start = time.perf_counter()
    model = load_condition(condition, config.runtime_path, spec.weights_path,
                           cast(Literal["causal", "noncausal"], spec.attention),
                           device="cuda" if cuda else "cpu")
    load_s = time.perf_counter() - load_start
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    visual_weights_equal = None
    if spec.weights_path is not None:
        from safetensors.torch import load_file

        runtime_visual = {name: tensor for name, tensor in model.runtime.backbone.state_dict().items()
                          if name.startswith("visual.")}
        base_visual = {}
        for shard in sorted(spec.weights_path.glob("*.safetensors")):
            base_visual.update({name: tensor for name, tensor in rename_base_weights(load_file(str(shard), device="cpu")).items()
                                if name.startswith("visual.")})
        visual_weights_equal = runtime_visual.keys() == base_visual.keys() and all(
            torch.equal(tensor.cpu(), base_visual[name]) for name, tensor in runtime_visual.items())
        del runtime_visual, base_visual

    def run_window(row: WindowRow) -> WindowResult:
        images = input_form.build(row.item, row.window, config.question).images
        return model.run_window(images, config.question)

    for name, planned in sets.items():
        extract(planned, run_window, out / name)
    write_run_json(out, config, condition, {"started": started, "finished": datetime.now(timezone.utc).isoformat(),
                                            "free_memory_gb": memory,
                                            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
                                            "load_s": load_s, "windows": {name: len(rows) for name, rows in sets.items()},
                                            "limit_windows": limit_windows,
                                            "visual_weights_equal": visual_weights_equal})
    return out
