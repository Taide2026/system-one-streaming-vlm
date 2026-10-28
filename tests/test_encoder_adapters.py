"""Tests for s1stream.encoder.adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest
import torch
import transformers as T
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from s1stream.encoder.adapters import (
    Dinov3Adapter,
    InternVitAdapter,
    QwenVitAdapter,
    SiglipAdapter,
    Vjepa2Adapter,
    load_encoder,
)


@pytest.fixture
def frames() -> list[Image.Image]:
    return [
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (48, 64, 3), dtype=np.uint8))
        for i in range(8)
    ]


@pytest.fixture
def qwen_pair() -> tuple[torch.nn.Module, Any]:
    torch.manual_seed(0)
    config = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=48,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=64,
    )
    model = Qwen3_5VisionModel._from_config(config).eval()
    processor = T.Qwen2VLImageProcessor(
        size={"shortest_edge": 32 * 32, "longest_edge": 64 * 64},
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
    )
    return model, processor


@pytest.fixture
def internvl_pair() -> tuple[torch.nn.Module, Any]:
    torch.manual_seed(0)
    vc = T.InternVLVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=[28, 28],
        patch_size=[14, 14],
    )
    tc = T.Qwen2Config(
        hidden_size=48,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=100,
    )
    model = T.InternVLModel(T.InternVLConfig(vision_config=vc, text_config=tc, downsample_ratio=0.5)).eval()
    processor = T.GotOcr2ImageProcessor(size={"height": 28, "width": 28}, crop_to_patches=False)
    return model, processor


@pytest.fixture
def siglip_pair() -> tuple[torch.nn.Module, Any]:
    torch.manual_seed(0)
    config = T.SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
    )
    model = T.SiglipVisionModel(config).eval()
    processor = T.SiglipImageProcessor(size={"height": 28, "width": 28})
    return model, processor


@pytest.fixture
def dinov3_pair() -> tuple[torch.nn.Module, Any]:
    torch.manual_seed(0)
    config = T.DINOv3ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=32,
        patch_size=16,
        num_register_tokens=4,
    )
    model = T.DINOv3ViTModel(config).eval()
    processor = T.DINOv3ViTImageProcessor(size={"height": 32, "width": 32})
    return model, processor


@pytest.fixture
def vjepa2_pair() -> tuple[torch.nn.Module, Any]:
    torch.manual_seed(0)
    config = T.VJEPA2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        crop_size=32,
        patch_size=16,
        tubelet_size=2,
        frames_per_clip=8,
        pred_hidden_size=32,
        pred_num_hidden_layers=1,
        pred_num_attention_heads=2,
        pred_mlp_ratio=2.0,
    )
    model = T.VJEPA2Model(config).eval()
    processor = T.VJEPA2VideoProcessor(size={"shortest_edge": 36}, crop_size={"height": 32, "width": 32})
    return model, processor


def test_qwen35_vit_kinds_and_last_state_mean(
    frames: list[Image.Image],
    qwen_pair: tuple[torch.nn.Module, Any],
) -> None:
    model, processor = qwen_pair
    adapter = QwenVitAdapter(model, processor)
    out = adapter.encode(frames)
    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        ref_last = model(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]).last_hidden_state.float()
    assert adapter.n_states == 3
    assert set(out.kinds.keys()) == {"patch_mean", "patch_max"}
    assert out.head == {}
    assert out.kinds["patch_mean"].shape == (3, 32)
    assert out.kinds["patch_max"].shape == (3, 32)
    assert torch.allclose(out.kinds["patch_mean"][-1], ref_last.mean(dim=0), atol=1e-5)
    assert torch.allclose(out.kinds["patch_max"][-1], ref_last.amax(dim=0), atol=1e-5)


def test_internvit_300m_states_match_vision_tower_hidden_states(
    frames: list[Image.Image],
    internvl_pair: tuple[torch.nn.Module, Any],
) -> None:
    model, processor = internvl_pair
    adapter = InternVitAdapter(model, processor)
    out = adapter.encode(frames)
    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        vision_out = model.vision_tower(inputs["pixel_values"], output_hidden_states=True)
    hidden_states = vision_out.hidden_states
    assert adapter.n_states == 3
    assert len(hidden_states) == 3
    assert set(out.kinds.keys()) == {"patch_mean", "patch_max", "cls_mean"}
    for kind in ("patch_mean", "patch_max", "cls_mean"):
        assert out.kinds[kind].shape == (3, 32)
    for s, state in enumerate(hidden_states):
        hs = state.float()
        assert torch.allclose(out.kinds["patch_mean"][s], hs[:, 1:].mean(dim=(0, 1)), atol=1e-5)
        assert torch.allclose(out.kinds["patch_max"][s], hs[:, 1:].amax(dim=(0, 1)), atol=1e-5)
        assert torch.allclose(out.kinds["cls_mean"][s], hs[:, 0].mean(dim=0), atol=1e-5)


def test_internvit_300m_head_matches_projector_stats(
    frames: list[Image.Image],
    internvl_pair: tuple[torch.nn.Module, Any],
) -> None:
    model, processor = internvl_pair
    adapter = InternVitAdapter(model, processor)
    out = adapter.encode(frames)
    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        feat_out = model.get_image_features(
            inputs["pixel_values"],
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
        )
    proj = feat_out.pooler_output.float()
    assert set(out.head.keys()) == {"patch_mean", "patch_max"}
    assert out.head["patch_mean"].shape == (1, 48)
    assert out.head["patch_max"].shape == (1, 48)
    assert torch.allclose(out.head["patch_mean"], proj.mean(dim=(0, 1)).unsqueeze(0), atol=1e-5)
    assert torch.allclose(out.head["patch_max"], proj.amax(dim=(0, 1)).unsqueeze(0), atol=1e-5)


def test_siglip2_so400m_states_and_pooler_head(
    frames: list[Image.Image],
    siglip_pair: tuple[torch.nn.Module, Any],
) -> None:
    model, processor = siglip_pair
    adapter = SiglipAdapter(model, processor)
    out = adapter.encode(frames)
    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        ref = model(inputs["pixel_values"], output_hidden_states=True)
    assert adapter.n_states == 3
    assert set(out.kinds.keys()) == {"patch_mean", "patch_max"}
    assert set(out.head.keys()) == {"pooler_mean"}
    for s, state in enumerate(ref.hidden_states):
        hs = state.float()
        assert torch.allclose(out.kinds["patch_mean"][s], hs.mean(dim=(0, 1)), atol=1e-5)
        assert torch.allclose(out.kinds["patch_max"][s], hs.amax(dim=(0, 1)), atol=1e-5)
    assert out.head["pooler_mean"].shape == (1, 32)
    expected_head = ref.pooler_output.float().mean(dim=0, keepdim=True)
    assert torch.allclose(out.head["pooler_mean"], expected_head, atol=1e-5)


def test_dinov3_vitl_skips_cls_and_registers_and_pools_cls(
    frames: list[Image.Image],
    dinov3_pair: tuple[torch.nn.Module, Any],
) -> None:
    model, processor = dinov3_pair
    adapter = Dinov3Adapter(model, processor)
    out = adapter.encode(frames)
    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        ref = model(inputs["pixel_values"], output_hidden_states=True)
    prefix = 1 + model.config.num_register_tokens
    assert prefix == 5
    assert adapter.n_states == 3
    assert set(out.kinds.keys()) == {"patch_mean", "patch_max", "cls_mean"}
    assert out.head == {}
    for s, state in enumerate(ref.hidden_states):
        hs = state.float()
        assert hs.shape == (8, 9, 32)
        assert torch.allclose(out.kinds["patch_mean"][s], hs[:, prefix:].mean(dim=(0, 1)), atol=1e-5)
        assert torch.allclose(out.kinds["patch_max"][s], hs[:, prefix:].amax(dim=(0, 1)), atol=1e-5)
        assert torch.allclose(out.kinds["cls_mean"][s], hs[:, 0].mean(dim=0), atol=1e-5)


def test_vjepa2_vitl_single_clip_tokens_and_states(
    frames: list[Image.Image],
    vjepa2_pair: tuple[torch.nn.Module, Any],
) -> None:
    model, processor = vjepa2_pair
    seen_shapes: list[tuple[int, ...]] = []

    def record_input(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if args:
            videos = args[0]
        else:
            videos = kwargs["pixel_values_videos"]
        seen_shapes.append(tuple(videos.shape))

    hook = model.register_forward_pre_hook(record_input, with_kwargs=True)
    try:
        adapter = Vjepa2Adapter(model, processor)
        out = adapter.encode(frames)
    finally:
        hook.remove()
    assert seen_shapes == [(1, 8, 3, 32, 32)]
    inputs = processor(videos=[frames], return_tensors="pt")
    with torch.no_grad():
        ref = model(
            pixel_values_videos=inputs["pixel_values_videos"],
            skip_predictor=True,
            output_hidden_states=True,
        )
    expected_tokens = (8 // model.config.tubelet_size) * (model.config.crop_size // model.config.patch_size) ** 2
    assert expected_tokens == 16
    assert adapter.n_states == 3
    assert set(out.kinds.keys()) == {"patch_mean", "patch_max"}
    assert out.head == {}
    for s, state in enumerate(ref.hidden_states):
        hs = state.float()
        assert hs.shape == (1, expected_tokens, 32)
        assert torch.allclose(out.kinds["patch_mean"][s], hs.mean(dim=(0, 1)), atol=1e-5)
        assert torch.allclose(out.kinds["patch_max"][s], hs.amax(dim=(0, 1)), atol=1e-5)


def test_all_adapters_disable_grad_and_return_detached_tensors(
    frames: list[Image.Image],
    qwen_pair: tuple[torch.nn.Module, Any],
    internvl_pair: tuple[torch.nn.Module, Any],
    siglip_pair: tuple[torch.nn.Module, Any],
    dinov3_pair: tuple[torch.nn.Module, Any],
    vjepa2_pair: tuple[torch.nn.Module, Any],
) -> None:
    cases = [
        (QwenVitAdapter, qwen_pair[0], qwen_pair[1], qwen_pair[0]),
        (InternVitAdapter, internvl_pair[0], internvl_pair[1], internvl_pair[0].vision_tower),
        (SiglipAdapter, siglip_pair[0], siglip_pair[1], siglip_pair[0]),
        (Dinov3Adapter, dinov3_pair[0], dinov3_pair[1], dinov3_pair[0]),
        (Vjepa2Adapter, vjepa2_pair[0], vjepa2_pair[1], vjepa2_pair[0]),
    ]
    for cls, model, processor, hooked_module in cases:
        grad_enabled: list[bool] = []

        def record_grad(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            grad_enabled.append(torch.is_grad_enabled())

        hook = hooked_module.register_forward_pre_hook(record_grad)
        try:
            adapter = cls(model, processor)
            out = adapter.encode(frames)
        finally:
            hook.remove()
        assert grad_enabled
        assert not any(grad_enabled)
        for tensor in list(out.kinds.values()) + list(out.head.values()):
            assert tensor.grad_fn is None
            assert not tensor.requires_grad
            assert tensor.dtype == torch.float32
            assert tensor.device.type == "cpu"


def test_load_encoder_unknown_name_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nope"):
        load_encoder("nope", tmp_path, "cpu")
