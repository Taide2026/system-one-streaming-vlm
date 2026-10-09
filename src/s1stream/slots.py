"""The five slots of an experiment. A plugin implements one protocol and registers under a name.

Flow per window: Dataset -> plan_windows -> InputForm (V1) -> Backbone (V2) -> AnswerHead (V3) -> score.
DecisionRule runs offline on the saved scores of one video.
"""
from __future__ import annotations

from typing import Protocol

from s1stream.types import BackboneOutput, ModelInput, Question, VideoItem, Window


class Dataset(Protocol):
    def items(self) -> list[VideoItem]:
        """Return the videos of the experiment, in a fixed order."""
        ...


class InputForm(Protocol):
    def build(self, item: VideoItem, window: Window, question: Question) -> ModelInput:
        """Turn one window into model input (frames, a caption, labels)."""
        ...


class Backbone(Protocol):
    def forward(self, inp: ModelInput, question: Question) -> BackboneOutput:
        """Run one forward pass without gradients. Return detached CPU tensors."""
        ...


class AnswerHead(Protocol):
    def probabilities(self, out: BackboneOutput, question: Question) -> dict[str, float]:
        """Return one probability per option of the question. The values sum to 1."""
        ...


class DecisionRule(Protocol):
    def settings(self) -> list[float]:
        """Return the parameter value of each alarm sequence, e.g. the thresholds."""
        ...

    def alarms(self, scores: list[float]) -> list[list[bool]]:
        """Return one alarm sequence per setting. Each sequence has one bool per window of one video."""
        ...
