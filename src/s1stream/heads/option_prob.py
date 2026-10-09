"""Answer head "option_prob" (V3): the Jev-Omni decision head on the backbone's hidden state.

Head256 is copied from jev_omni.py `_Head256` (revision 5addda8): normalise the hidden state with the stored
mean and standard deviation, then one linear logit per option slot (256 slots); softmax over the first n slots.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from s1stream.registry import ANSWER_HEADS
from s1stream.types import BackboneOutput, Question


class Head256(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.register_buffer("mu", torch.zeros(1, hidden))
        self.register_buffer("sd", torch.ones(1, hidden))
        self.linear = torch.nn.Linear(hidden, 256, dtype=torch.float32)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear((features.float() - self.mu) / self.sd)


@ANSWER_HEADS.register("option_prob")
class OptionProb:
    """Config: {name: option_prob, asset: jev-omni}. load_config adds path (the Jev-Omni folder).

    The head weights are path / "head.pt"; the hidden size is decision_config.json["hidden_size"].
    """

    def __init__(self, path: str | Path, asset: str | None = None) -> None:
        """Load Head256 on the CPU with torch.load(..., map_location="cpu", weights_only=True); eval mode."""
        folder = Path(path)
        hidden = json.loads((folder / "decision_config.json").read_text())["hidden_size"]
        self.head = Head256(hidden)
        self.head.load_state_dict(torch.load(folder / "head.pt", map_location="cpu", weights_only=True))
        self.head.eval()

    @classmethod
    def from_module(cls, head: Head256) -> "OptionProb":
        """Build from an existing head (used by tests)."""
        instance = cls.__new__(cls)
        instance.head = head
        return instance

    def probabilities(self, out: BackboneOutput, question: Question) -> dict[str, float]:
        """Raise ValueError when question has fewer than 2 or more than 256 options.

        Under torch.inference_mode(): logits = head(out.hidden)[0, :n]; return {option: softmax value} in option order.
        """
        n = len(question.options)
        if not 2 <= n <= 256:
            raise ValueError("Jev-Omni needs 2–256 options")
        with torch.inference_mode():
            logits = self.head(out.hidden)[0, :n]
            values = logits.softmax(dim=-1).tolist()
        return dict(zip(question.options, values))
