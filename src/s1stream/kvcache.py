"""Bookkeeping of a KV cache kept across windows (context strategies kv, kv_sinks, kv_compress).

The cache (transformers DynamicCache) holds one entry per layer. Full-attention layers keep keys and values per token
(.keys / .values of shape [batch, heads, tokens, dim]); linear-attention layers (Gated DeltaNet, 3 in 4 layers of
Qwen3.5) keep a fixed-size conv + recurrent state that has seen every token so far and cannot forget one. Evicting a
window therefore removes its tokens from the attention layers and from our token list; the recurrent state keeps a
trace of it. Positions run on: an evicted frame keeps its place in the position index (history), as in a stream.

Segments: the cache's tokens are  prefix | frames of window a | frames of window b | ... ; one Segment per block, with
its tokens' ids and hidden states (the head reads the whole sequence as memory). The question's tokens are never kept.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch

from s1stream.context import CachePolicy


@dataclass
class Segment:
    """One block of cached tokens. window is None for the prompt prefix."""
    window: int | None
    ids: list[int]
    hidden: Any  # tensor (tokens, hidden_size), on the model's device
    n_images: int = 0

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class StreamState:
    """The cache of one video and what is in it."""
    cache: Any
    segments: list[Segment] = field(default_factory=list)
    history_ids: list[int] = field(default_factory=list)  # every token ever prefilled, evicted ones too (positions)
    history_grid: list[Any] = field(default_factory=list)  # one grid_thw row per image ever prefilled

    def cached_ids(self) -> list[int]:
        return [token for segment in self.segments for token in segment.ids]

    def cached_hidden(self) -> Any:
        return torch.cat([segment.hidden for segment in self.segments], dim=0)

    def frame_tokens(self) -> int:
        return sum(len(segment) for segment in self.segments if segment.window is not None)


def snapshot(cache: Any) -> Any:
    """A deep copy of the cache: restore by using the copy."""
    return copy.deepcopy(cache)


def remove_tokens(cache: Any, indices: list[int]) -> None:
    """Drop the tokens at these positions (in cache order) from every attention layer that holds keys and values.
    Linear-attention layers have no per-token state and are left alone."""
    if not indices:
        return
    drop = set(indices)
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        if keys is None or not isinstance(keys, torch.Tensor) or keys.numel() == 0:
            continue
        keep = torch.tensor([i for i in range(keys.shape[-2]) if i not in drop], device=keys.device, dtype=torch.long)
        layer.keys = keys.index_select(-2, keep)
        layer.values = layer.values.index_select(-2, keep)


def evict(state: StreamState, policy: CachePolicy, current_window: int) -> list[int]:
    """Before window current_window's frames go in: drop the frame segments older than keep_windows, except the sinks.
    Return the windows evicted."""
    oldest_kept = current_window - policy.keep_windows
    gone = [segment for segment in state.segments
            if segment.window is not None and segment.window < oldest_kept and segment.window >= policy.sink_windows]
    if not gone:
        return []
    offsets = _offsets(state.segments)
    indices = [offsets[i] + k for i, segment in enumerate(state.segments) if segment in gone for k in range(len(segment))]
    remove_tokens(state.cache, indices)
    state.segments = [segment for segment in state.segments if segment not in gone]
    return [segment.window for segment in gone]


def compress(state: StreamState, budget_tokens: int) -> int:
    """kv_compress: while the frame segments hold more than budget_tokens tokens, keep every other token of the
    oldest frame segment that still has more than 2 tokens. Return the number of tokens removed."""
    removed = 0
    while state.frame_tokens() > budget_tokens:
        target = next((i for i, segment in enumerate(state.segments) if segment.window is not None and len(segment) > 2), None)
        if target is None:
            break
        segment = state.segments[target]
        offset = _offsets(state.segments)[target]
        keep = list(range(0, len(segment), 2))
        drop = [offset + k for k in range(len(segment)) if k % 2 == 1]
        remove_tokens(state.cache, drop)
        segment.ids = [segment.ids[k] for k in keep]
        segment.hidden = segment.hidden[keep]
        removed += len(drop)
    return removed


def _offsets(segments: list[Segment]) -> list[int]:
    offsets, total = [], 0
    for segment in segments:
        offsets.append(total)
        total += len(segment)
    return offsets
