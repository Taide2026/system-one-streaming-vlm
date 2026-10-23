"""Unit kvcache: eviction, compression and token removal on a fake cache. No model."""
import torch

from s1stream.context import CachePolicy
from s1stream.kvcache import Segment, StreamState, compress, evict, remove_tokens, snapshot


class FakeAttentionLayer:
    def __init__(self, tokens: int):
        self.keys = torch.arange(tokens, dtype=torch.float32).view(1, 1, tokens, 1)
        self.values = self.keys.clone() * 10


class FakeLinearLayer:
    """A Gated DeltaNet layer: a state, no per-token keys."""
    def __init__(self):
        self.recurrent_states = {0: torch.zeros(2)}


class FakeCache:
    def __init__(self, tokens: int):
        self.layers = [FakeAttentionLayer(tokens), FakeLinearLayer()]


def segment(window, n, start):
    return Segment(window, list(range(start, start + n)), torch.arange(start, start + n, dtype=torch.float32).view(n, 1))


def state_with(windows: int, prefix: int = 3, per_window: int = 4) -> StreamState:
    state = StreamState(cache=None, segments=[segment(None, prefix, 0)])
    for w in range(windows):
        state.segments.append(segment(w, per_window, prefix + w * per_window))
    state.cache = FakeCache(prefix + windows * per_window)
    return state


def test_remove_tokens_drops_positions_from_attention_layers_only():
    cache = FakeCache(6)
    remove_tokens(cache, [1, 2])
    assert cache.layers[0].keys.flatten().tolist() == [0, 3, 4, 5]
    assert cache.layers[0].values.flatten().tolist() == [0, 30, 40, 50]
    assert cache.layers[1].recurrent_states[0].tolist() == [0, 0]


def test_evict_keeps_prefix_the_last_keep_windows_and_the_sinks():
    state = state_with(windows=3)  # windows 0, 1, 2 cached; window 3 is about to come in
    gone = evict(state, CachePolicy(keep_windows=1, sink_windows=1), current_window=3)
    assert gone == [1]
    assert [s.window for s in state.segments] == [None, 0, 2]
    # tokens of window 1 (positions 7..10) are gone from the attention layer; the cache order matches the segments
    assert state.cache.layers[0].keys.flatten().tolist() == state.cached_ids()


def test_evict_nothing_while_within_keep_windows():
    state = state_with(windows=2)
    assert evict(state, CachePolicy(keep_windows=2), current_window=2) == []
    assert len(state.segments) == 3


def test_compress_halves_the_oldest_frame_segment_until_within_budget():
    state = state_with(windows=3, prefix=2, per_window=8)  # 24 frame tokens
    removed = compress(state, budget_tokens=16)
    assert removed == 8 and state.frame_tokens() == 16
    assert len(state.segments[1]) == 4 and state.segments[1].ids == [2, 4, 6, 8]  # every other token of window 0
    assert state.segments[1].hidden.flatten().tolist() == [2, 4, 6, 8]
    assert state.cache.layers[0].keys.flatten().tolist() == state.cached_ids()
    assert len(state.segments[2]) == 8  # untouched


def test_compress_leaves_segments_of_two_tokens():
    state = state_with(windows=1, prefix=1, per_window=2)
    assert compress(state, budget_tokens=1) == 0


def test_snapshot_is_independent_of_the_live_cache():
    cache = FakeCache(4)
    saved = snapshot(cache)
    remove_tokens(cache, [0])
    assert saved.layers[0].keys.shape[-2] == 4 and cache.layers[0].keys.shape[-2] == 3
