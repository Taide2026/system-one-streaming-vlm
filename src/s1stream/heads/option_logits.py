"""Answer head "option_logits" (V3): softmax over per-option logits that the backbone already computed.

For backbones whose model scores the options itself (clef_flash, imajev): BackboneOutput.logits holds one logit per
option of the question, in question.options order.
"""
from __future__ import annotations

import torch

from s1stream.registry import ANSWER_HEADS
from s1stream.types import BackboneOutput, Question


@ANSWER_HEADS.register("option_logits")
class OptionLogits:
    """Config: {name: option_logits}."""

    def __init__(self, asset: str | None = None, path: str | None = None) -> None:
        pass

    def probabilities(self, out: BackboneOutput, question: Question) -> dict[str, float]:
        """Raise ValueError when out.logits is None or does not hold one logit per option.

        Return {option: softmax(out.logits)} in option order.
        """
        if out.logits is None:
            raise ValueError("The backbone returned no option logits")
        logits = torch.as_tensor(out.logits, dtype=torch.float32).flatten()
        if logits.numel() != len(question.options):
            raise ValueError(f"Expected {len(question.options)} option logits, got {logits.numel()}")
        return dict(zip(question.options, logits.softmax(dim=-1).tolist()))
