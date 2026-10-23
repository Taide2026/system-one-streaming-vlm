"""The five slots of an experiment. A plugin implements one protocol and registers under a name.

Flow per window: Dataset -> plan_windows -> InputForm (V1) -> Backbone (V2) -> AnswerHead (V3) -> score.
DecisionRule runs offline on the saved scores of one video.
"""
from __future__ import annotations

from typing import Protocol

from s1stream.context import CachePolicy
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


class StreamingBackbone(Backbone, Protocol):
    """A causal backbone that can keep the past across the windows of one video (context strategies kv*)."""

    def start_video(self) -> None:
        """Forget everything: the next forward_stream starts a new video."""
        ...

    def forward_stream(self, inp: ModelInput, question: Question, policy: CachePolicy, window: int) -> BackboneOutput:
        """Prefill only inp.images (the newest window's frames) on top of the kept cache, append the question, read
        the option logits, then drop the question's tokens. Keep the past as policy says."""
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
