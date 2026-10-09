"""Read an experiment config (YAML) into an ExperimentConfig.

Example: configs/jev_omni_ucf20.yaml. A config never holds an absolute path.
A slot spec may name an asset: {"name": "jev_omni", "asset": "jev-omni"}. load_config then adds
"path": <resolved folder> to that spec (see s1stream.assets.resolve) and keeps "asset".
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from s1stream.types import Question

SECTIONS = ("name", "dataset", "window", "question", "input_form", "backbone", "answer_head", "decision_rules")
SLOT_SECTIONS = ("dataset", "input_form", "backbone", "answer_head")


class ConfigError(Exception):
    """The config is invalid. The message names the section or the field."""


@dataclass(frozen=True)
class WindowSpec:
    length_s: float
    stride_s: float
    frames: int


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: dict[str, Any]
    window: WindowSpec
    question: Question
    input_form: dict[str, Any]
    backbone: dict[str, Any]
    answer_head: dict[str, Any]
    decision_rules: list[dict[str, Any]]
    raw: dict[str, Any]  # the YAML as read, before asset resolution; copied into each run folder


def load_config(path: Path, resolver: Callable[[str], Path] | None = None) -> ExperimentConfig:
    """Read and check the config at path.

    - Raise ConfigError naming the section when one of SECTIONS is missing.
    - Raise ConfigError when question.positive is not in question.options.
    - For each spec in SLOT_SECTIONS and in decision_rules that has the key "asset", set
      spec["path"] = resolver(spec["asset"]). resolver defaults to s1stream.assets.resolve.
      An s1stream.assets.AssetError from the resolver becomes a ConfigError that names the asset.
    - question.options becomes a tuple.
    """
    import yaml

    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read config: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Config must be a mapping")
    for section in SECTIONS:
        if section not in data:
            raise ConfigError(f"Missing section: {section}")

    question_data = data["question"]
    if question_data["positive"] not in question_data["options"]:
        raise ConfigError("question.positive must be in question.options")

    original = copy.deepcopy(data)
    specs = [data[section] for section in SLOT_SECTIONS]
    specs.extend(data["decision_rules"])
    asset_specs = [spec for spec in specs if "asset" in spec]
    if asset_specs:
        from s1stream.assets import AssetError

        if resolver is None:
            from s1stream.assets import resolve

            resolver = resolve
        for spec in asset_specs:
            try:
                spec["path"] = resolver(spec["asset"])
            except AssetError as exc:
                raise ConfigError(f"Could not resolve asset {spec['asset']!r}: {exc}") from exc

    window = data["window"]
    return ExperimentConfig(
        name=data["name"],
        dataset=data["dataset"],
        window=WindowSpec(**window),
        question=Question(
            state=question_data["state"],
            text=question_data["text"],
            options=tuple(question_data["options"]),
            positive=question_data["positive"],
        ),
        input_form=data["input_form"],
        backbone=data["backbone"],
        answer_head=data["answer_head"],
        decision_rules=data["decision_rules"],
        raw=original,
    )
