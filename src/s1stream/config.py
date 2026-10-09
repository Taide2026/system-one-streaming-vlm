"""Read an experiment config (YAML) into an ExperimentConfig.

Example: configs/jev_omni_ucf20.yaml. A config never holds an absolute path.
A slot spec may name an asset: {"name": "jev_omni", "asset": "jev-omni"}. load_config then adds
"path": <resolved folder> to that spec (see s1stream.assets.resolve) and keeps "asset".
"""
from __future__ import annotations

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
    raise NotImplementedError
