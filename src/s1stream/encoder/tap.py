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
        self.n_states = len(blocks) + 1
        self._states: list[torch.Tensor] = []
        self._hooks = []
        if blocks:
            self._hooks.append(blocks[0].register_forward_pre_hook(self._pre, with_kwargs=True))
            self._hooks.extend(block.register_forward_hook(self._block) for block in blocks)

    def _pre(self, module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = args[0] if args else kwargs["hidden_states"]
        self._states.append(hidden.detach())

    def _block(self, module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self._states.append(hidden.detach())

    def states(self) -> list[torch.Tensor]:
        """The recorded states in order. Raise RuntimeError naming both counts when the count is not n_states."""
        if len(self._states) != self.n_states:
            raise RuntimeError(f"expected {self.n_states} states, got {len(self._states)}")
        return list(self._states)

    def clear(self) -> None:
        """Forget the recorded states."""
        self._states.clear()

    def close(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self) -> StateTap:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def pool(states: Sequence[torch.Tensor], prefix: int, cls: bool) -> dict[str, torch.Tensor]:
    """Pool the states of one window into one vector per state and kind.

    Each state has shape (frames, tokens, dims), or (tokens, dims) which is treated as 1 frame.
    Patch tokens are tokens prefix: of every frame (CLS and register tokens come first).
    Returns {"patch_mean": mean over frames and patch tokens, "patch_max": max over frames and patch tokens}
    plus "cls_mean" (mean over frames of token 0) when cls is True.
    Each value: float32 CPU tensor of shape (len(states), dims), without grad_fn.
    """
    pooled: dict[str, list[torch.Tensor]] = {"patch_mean": [], "patch_max": []}
    if cls:
        pooled["cls_mean"] = []
    for state in states:
        hidden = state.detach().to(dtype=torch.float32)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        patches = hidden[:, prefix:, :]
        pooled["patch_mean"].append(patches.mean(dim=(0, 1)).cpu())
        pooled["patch_max"].append(patches.amax(dim=(0, 1)).cpu())
        if cls:
            pooled["cls_mean"].append(hidden[:, 0, :].mean(dim=0).cpu())
    return {key: torch.stack(values) for key, values in pooled.items()}
