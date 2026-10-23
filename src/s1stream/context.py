"""The context strategies: how the past enters a 2 s decision (config section `context`, optional; default s0).

The decision and its label stay on the newest window in every strategy, so every run is scored on the same frames.

  s0           today: the newest window only, nothing kept.
  recompute    S1: the newest window + the frames of the past L s, read again every window (the upper bound of what
               more past buys; costs (L / stride + 1) x frames per window).
  kv           S2: the KV cache / recurrent state of the past L s stays on the GPU; a window prefills only its new
               frames; the question is appended, read, then dropped (snapshot -> restore). Needs a StreamingBackbone.
  kv_sinks     S3: kv, plus the frames of the first sinks_s seconds of the video that are never evicted (StreamingVLM's
               attention sinks; the prompt prefix is always kept in every kv strategy).
  kv_compress  S4: kv, but the past's visual tokens are pruned to budget_tokens (oldest window first, every other
               token), so memory stays flat as L grows (a stand-in for InfiniPot-V / StreamMem's scored pruning).
  text         S5: nothing cached; the abnormal probabilities of the last `history` windows are written into the
               question's state as text (MoniTor-like memory). Works with every backbone.

Config:
  context: {strategy: kv, past_s: 2.0, sinks_s: 2.0, budget_tokens: 640, history: 4}
past_s is a multiple of window.stride_s; keep_windows = past_s / stride_s past windows are kept or re-read.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from s1stream.types import Question, Window

STRATEGIES = ("s0", "recompute", "kv", "kv_sinks", "kv_compress", "text")
KV_STRATEGIES = ("kv", "kv_sinks", "kv_compress")


class ContextError(Exception):
    """The context section is invalid. The message names the field."""


@dataclass(frozen=True)
class ContextSpec:
    strategy: str = "s0"
    past_s: float = 0.0
    sinks_s: float = 0.0
    budget_tokens: int | None = None
    history: int = 0

    def keep_windows(self, stride_s: float) -> int:
        """How many past windows the model sees (recompute) or keeps (kv*): past_s / stride_s, rounded."""
        return int(round(self.past_s / stride_s))

    def sink_windows(self, stride_s: float) -> int:
        return int(round(self.sinks_s / stride_s))


@dataclass(frozen=True)
class CachePolicy:
    """What a StreamingBackbone keeps after each window.

    keep_windows: past windows whose frames stay in the cache (the newest window is always kept).
    sink_windows: the first windows of the video that are never evicted.
    budget_tokens: when set, the past's visual tokens (sinks included) are pruned down to this many.
    """
    keep_windows: int
    sink_windows: int = 0
    budget_tokens: int | None = None


def parse_context(data: dict[str, Any] | None, stride_s: float) -> ContextSpec:
    """Read the `context` section. None -> s0. Raise ContextError for an unknown strategy, a past_s that is not a
    multiple of stride_s, a kv_compress without budget_tokens, or a text without history."""
    if data is None:
        return ContextSpec()
    strategy = data.get("strategy", "s0")
    if strategy not in STRATEGIES:
        raise ContextError(f"Unknown context.strategy {strategy!r}: one of {', '.join(STRATEGIES)}")
    spec = ContextSpec(strategy=strategy, past_s=float(data.get("past_s", 0.0)), sinks_s=float(data.get("sinks_s", 0.0)),
                       budget_tokens=data.get("budget_tokens"), history=int(data.get("history", 0)))
    if strategy in ("recompute", *KV_STRATEGIES):
        windows = spec.past_s / stride_s
        if spec.past_s <= 0 or abs(windows - round(windows)) > 1e-6:
            raise ContextError("context.past_s must be a positive multiple of window.stride_s")
    if strategy == "kv_sinks" and spec.sinks_s <= 0:
        raise ContextError("context.sinks_s must be positive for kv_sinks")
    if strategy == "kv_compress" and not spec.budget_tokens:
        raise ContextError("context.budget_tokens is required for kv_compress")
    if strategy == "text" and spec.history <= 0:
        raise ContextError("context.history must be positive for text")
    return spec


def policy(spec: ContextSpec, stride_s: float) -> CachePolicy:
    """The CachePolicy of a kv strategy."""
    return CachePolicy(keep_windows=spec.keep_windows(stride_s),
                       sink_windows=spec.sink_windows(stride_s) if spec.strategy == "kv_sinks" else 0,
                       budget_tokens=spec.budget_tokens if spec.strategy == "kv_compress" else None)


def with_past_frames(windows: list[Window], index: int, keep_windows: int) -> Window:
    """recompute: windows[index] with the frames of the past keep_windows windows put before its own."""
    past = windows[max(0, index - keep_windows):index]
    frames = tuple(frame for window in past for frame in window.frame_indices) + windows[index].frame_indices
    return replace(windows[index], frame_indices=frames)


def text_state(question: Question, scores: list[float], history: int) -> Question:
    """text: the question whose state ends with the abnormal probabilities of the last `history` windows, oldest
    first; unchanged when there is no past yet."""
    past = scores[-history:]
    if not past:
        return question
    listed = ", ".join(f"{score:.2f}" for score in past)
    return replace(question, state=f"{question.state} Abnormal probability of the previous windows, oldest first: {listed}.")
