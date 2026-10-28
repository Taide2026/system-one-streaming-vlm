"""Tests for s1stream.encoder.tap (StateTap and pool)."""
from __future__ import annotations

import pytest
import torch

from s1stream.encoder.tap import StateTap, pool


class _ScaleAddBlock(torch.nn.Module):
    def __init__(self, scale: float, bias: float) -> None:
        super().__init__()
        self.scale = scale
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


class _TupleBlock(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x + 1.0, None


class _KeywordBlock(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 3.0


def test_state_tap_records_input_and_each_block_output() -> None:
    blocks = [_ScaleAddBlock(2.0, 1.0), _ScaleAddBlock(3.0, -2.0), _ScaleAddBlock(0.5, 4.0)]
    x0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with StateTap(blocks) as tap:
        assert tap.n_states == 4
        x = x0
        expected = [x0]
        for block in blocks:
            x = block(x)
            expected.append(x)
        recorded = tap.states()
    assert len(recorded) == 4
    for actual, target in zip(recorded, expected, strict=True):
        assert torch.equal(actual, target)


def test_state_tap_unwraps_tuple_block_output() -> None:
    blocks: list[torch.nn.Module] = [_TupleBlock(), _ScaleAddBlock(2.0, 0.0)]
    x0 = torch.tensor([[1.0, 5.0]])
    with StateTap(blocks) as tap:
        hidden, _ = blocks[0](x0)
        out = blocks[1](hidden)
        recorded = tap.states()
    assert len(recorded) == 3
    assert torch.equal(recorded[0], x0)
    assert torch.equal(recorded[1], hidden)
    assert torch.equal(recorded[2], out)


def test_state_tap_captures_keyword_hidden_states_on_first_block() -> None:
    blocks: list[torch.nn.Module] = [_KeywordBlock(), _ScaleAddBlock(2.0, 1.0)]
    x0 = torch.tensor([[2.0, 4.0]])
    with StateTap(blocks) as tap:
        h1 = blocks[0](hidden_states=x0)
        h2 = blocks[1](h1)
        recorded = tap.states()
    assert len(recorded) == 3
    assert torch.equal(recorded[0], x0)
    assert torch.equal(recorded[1], h1)
    assert torch.equal(recorded[2], h2)


def test_state_tap_removes_hooks_on_context_exit() -> None:
    blocks = [_ScaleAddBlock(2.0, 1.0), _ScaleAddBlock(3.0, 0.0)]
    x0 = torch.tensor([[1.0, 2.0]])
    with StateTap(blocks) as tap:
        x = x0
        for block in blocks:
            x = block(x)
        before = list(tap.states())
    x = x0 + 10.0
    for block in blocks:
        x = block(x)
    after = tap.states()
    assert len(after) == len(before) == 3
    for first, second in zip(before, after, strict=True):
        assert torch.equal(first, second)


def test_state_tap_raises_after_two_forwards_without_clear() -> None:
    blocks = [_ScaleAddBlock(2.0, 1.0), _ScaleAddBlock(3.0, 0.0)]
    x0 = torch.tensor([[1.0, 2.0]])
    with StateTap(blocks) as tap:
        for _ in range(2):
            x = x0
            for block in blocks:
                x = block(x)
        with pytest.raises(RuntimeError) as exc_info:
            tap.states()
        assert not isinstance(exc_info.value, NotImplementedError)
        assert "3" in str(exc_info.value) and "6" in str(exc_info.value)


def test_pool_uses_only_tokens_after_prefix_for_patch_stats() -> None:
    state = torch.arange(2 * 9 * 4, dtype=torch.float32).reshape(2, 9, 4)
    state[:, :5, :] = 10000.0
    out = pool([state], prefix=5, cls=False)
    patches = state[:, 5:, :]
    expected_mean = patches.mean(dim=(0, 1)).unsqueeze(0)
    expected_max = patches.amax(dim=(0, 1)).unsqueeze(0)
    assert torch.allclose(out["patch_mean"], expected_mean)
    assert torch.allclose(out["patch_max"], expected_max)


def test_pool_cls_flag_controls_cls_mean_output() -> None:
    state = torch.arange(3 * 6 * 4, dtype=torch.float32).reshape(3, 6, 4)
    with_cls = pool([state], prefix=1, cls=True)
    without_cls = pool([state], prefix=1, cls=False)
    expected_cls = state[:, 0, :].mean(dim=0).unsqueeze(0)
    assert set(with_cls.keys()) == {"patch_mean", "patch_max", "cls_mean"}
    assert torch.allclose(with_cls["cls_mean"], expected_cls)
    assert set(without_cls.keys()) == {"patch_mean", "patch_max"}


def test_pool_treats_2d_state_as_single_frame() -> None:
    state_2d = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    out_2d = pool([state_2d], prefix=1, cls=True)
    out_3d = pool([state_2d.unsqueeze(0)], prefix=1, cls=True)
    assert out_2d["patch_mean"].shape == (1, 4)
    assert torch.allclose(out_2d["patch_mean"], state_2d[1:].mean(dim=0, keepdim=True))
    assert torch.allclose(out_2d["patch_max"], state_2d[1:].amax(dim=0, keepdim=True))
    assert torch.allclose(out_2d["cls_mean"], state_2d[0:1])
    for key in ("patch_mean", "patch_max", "cls_mean"):
        assert torch.equal(out_2d[key], out_3d[key])


def test_pool_returns_detached_float32_cpu_tensors() -> None:
    raw = [torch.randn(2, 5, 4, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    states = [tensor * 2.0 for tensor in raw]
    out = pool(states, prefix=1, cls=True)
    for key in ("patch_mean", "patch_max", "cls_mean"):
        value = out[key]
        assert value.shape == (3, 4)
        assert value.dtype == torch.float32
        assert value.device.type == "cpu"
        assert not value.requires_grad
        assert value.grad_fn is None
