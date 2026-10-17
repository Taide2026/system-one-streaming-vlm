"""Answer head "generated_answer" (V3): probabilities from a decoded answer (backbone gemma_generate).

- With option logits (constrained decoding): softmax over them, as option_logits; the decoded option is their argmax.
- Without (free decoding): parse BackboneOutput.text. The answer is option o when its first word is o, or else when
  o is the only option that appears in it as a word (case-insensitive). The answer gets probability 1, the others 0.
  An invalid answer (no option, or several) gets 0 for every option, so its score is 0: it never flags a window.
"""
from __future__ import annotations

import re

import torch

from s1stream.registry import ANSWER_HEADS
from s1stream.types import BackboneOutput, Question

WORD = re.compile(r"[a-z0-9]+")


def parse(text: str, options: tuple[str, ...]) -> str | None:
    """The option text answers (see the module docstring), or None when it is invalid."""
    words = WORD.findall(text.lower())
    lowered = {option.lower(): option for option in options}
    if words and words[0] in lowered:
        return lowered[words[0]]
    found = [option for key, option in lowered.items() if key in words]
    return found[0] if len(found) == 1 else None


@ANSWER_HEADS.register("generated_answer")
class GeneratedAnswer:
    """Config: {name: generated_answer}."""

    def __init__(self, asset: str | None = None, path: str | None = None) -> None:
        pass

    def probabilities(self, out: BackboneOutput, question: Question) -> dict[str, float]:
        """Raise ValueError when out has neither text nor option logits, or the wrong number of logits."""
        if out.logits is not None:
            logits = torch.as_tensor(out.logits, dtype=torch.float32).flatten()
            if logits.numel() != len(question.options):
                raise ValueError(f"Expected {len(question.options)} option logits, got {logits.numel()}")
            return dict(zip(question.options, logits.softmax(dim=-1).tolist()))
        if out.text is None:
            raise ValueError("The backbone returned neither an answer text nor option logits")
        answer = parse(out.text, question.options)
        return {option: float(option == answer) for option in question.options}
