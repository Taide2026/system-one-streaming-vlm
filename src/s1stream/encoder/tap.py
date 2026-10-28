"""Hidden-state tap and token pooling for vision encoders (encoder-probing decision 4)."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class StateTap:
    """Record the hidden state at every block boundary of a vision encoder.

    State 0 is the input of blocks[0] (pre-hook; the first positional argument, else the keyword "hidden_states").
    State s (1..len(blocks)) is the output of blocks[s - 1] (output[0] when the output is a tuple).
    Each state is stored with .detach() and is not moved or cast.
    Use as a context manager: hooks are registered in __init__ and removed in __exit__ (and by close()).
    """

    n_states: int  # len(blocks) + 1, set in __init__

    def __init__(self, blocks: Sequence[torch.nn.Module]) -> None:
        raise NotImplementedError

    def states(self) -> list[torch.Tensor]:
        """The recorded states in order. Raise RuntimeError naming both counts when the count is not n_states."""
        raise NotImplementedError

    def clear(self) -> None:
        """Forget the recorded states."""
        raise NotImplementedError

    def close(self) -> None:
        """Remove all hooks."""
        raise NotImplementedError

    def __enter__(self) -> StateTap:
        raise NotImplementedError

    def __exit__(self, *exc: Any) -> None:
        raise NotImplementedError


def pool(states: Sequence[torch.Tensor], prefix: int, cls: bool) -> dict[str, torch.Tensor]:
    """Pool the states of one window into one vector per state and kind.

    Each state has shape (frames, tokens, dims), or (tokens, dims) which is treated as 1 frame.
    Patch tokens are tokens prefix: of every frame (CLS and register tokens come first).
    Returns {"patch_mean": mean over frames and patch tokens, "patch_max": max over frames and patch tokens}
    plus "cls_mean" (mean over frames of token 0) when cls is True.
    Each value: float32 CPU tensor of shape (len(states), dims), without grad_fn.
    """
    raise NotImplementedError
