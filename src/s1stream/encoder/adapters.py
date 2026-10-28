"""Vision encoder adapters and loader (encoder-probing decisions 1, 3, 4)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from PIL import Image
import torch

from .tap import StateTap, pool


def _move_inputs(inputs: dict[str, torch.Tensor], model: torch.nn.Module, device: str) -> dict[str, torch.Tensor]:
    dtype = next(model.parameters()).dtype
    return {
        key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        for key, value in inputs.items()
    }


ENCODERS = ("qwen35-vit", "internvit-300m", "siglip2-so400m", "dinov3-vitl", "vjepa2-vitl")


@dataclass(frozen=True)
class EncoderOutput:
    """One window. kinds: {kind: float32 CPU tensor (n_states, dims)}; head: {kind: float32 CPU tensor (1, dims)},
    empty when the encoder has no head points. No tensor has a grad_fn."""

    kinds: dict[str, torch.Tensor]
    head: dict[str, torch.Tensor]


class QwenVitAdapter:
    """qwen35-vit: the Qwen3.5-9B vision tower (transformers Qwen3_5VisionModel), image input.
    model: Qwen3_5VisionModel; processor: its image processor (Qwen2VLImageProcessor).
    Each frame is one image. States: StateTap(model.blocks); no CLS, prefix 0. Kinds patch_mean, patch_max. No head."""

    name = "qwen35-vit"
    kinds = ("patch_mean", "patch_max")
    head_kinds = ()
    n_states: int

    def __init__(self, model: torch.nn.Module, processor: Any, device: str = "cpu") -> None:
        self.model: Any = model
        self.processor, self.device = processor, device
        self.n_states = len(self.model.blocks) + 1

    def encode(self, frames: list[Image.Image]) -> EncoderOutput:
        """Run under torch.inference_mode(). Inputs are moved to device and cast to the model dtype."""
        inputs = _move_inputs(self.processor(images=frames, return_tensors="pt"), self.model, self.device)
        with torch.inference_mode(), StateTap(self.model.blocks) as tap:
            self.model(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
            kinds = pool(tap.states(), prefix=0, cls=False)
        return EncoderOutput(kinds, {})


class InternVitAdapter:
    """internvit-300m: InternVLModel vision tower and multi-modal projector, image input.
    model: InternVLModel (its language_model may be absent or unused); processor: GotOcr2ImageProcessor with
    crop_to_patches=False (one tile per frame). States: StateTap(model.vision_tower.encoder.layer); token 0 is CLS,
    prefix 1. Kinds patch_mean, patch_max, cls_mean.
    Forward: model.get_image_features(pixel_values, vision_feature_layer=-1, vision_feature_select_strategy="default");
    its pooler_output (frames, tokens, text hidden) is the projector output.
    Head kinds patch_mean, patch_max over all frames and projector tokens, shape (1, text hidden)."""

    name = "internvit-300m"
    kinds = ("patch_mean", "patch_max", "cls_mean")
    head_kinds = ("patch_mean", "patch_max")
    n_states: int

    def __init__(self, model: torch.nn.Module, processor: Any, device: str = "cpu") -> None:
        self.model: Any = model
        self.processor, self.device = processor, device
        self.n_states = len(self.model.vision_tower.encoder.layer) + 1

    def encode(self, frames: list[Image.Image]) -> EncoderOutput:
        """Run under torch.inference_mode(). Inputs are moved to device and cast to the model dtype."""
        inputs = _move_inputs(self.processor(images=frames, return_tensors="pt"), self.model, self.device)
        with torch.inference_mode(), StateTap(self.model.vision_tower.encoder.layer) as tap:
            output = self.model.get_image_features(
                inputs["pixel_values"], vision_feature_layer=-1, vision_feature_select_strategy="default"
            )
            kinds = pool(tap.states(), prefix=1, cls=True)
            head = pool([output.pooler_output], prefix=0, cls=False)
        return EncoderOutput(kinds, head)


class SiglipAdapter:
    """siglip2-so400m: SiglipVisionModel, image input.
    model: SiglipVisionModel; processor: SiglipImageProcessor.
    States: StateTap(model.encoder.layers); prefix 0. Kinds patch_mean, patch_max.
    Head kind pooler_mean: mean over frames of the model output pooler_output, shape (1, hidden)."""

    name = "siglip2-so400m"
    kinds = ("patch_mean", "patch_max")
    head_kinds = ("pooler_mean",)
    n_states: int

    def __init__(self, model: torch.nn.Module, processor: Any, device: str = "cpu") -> None:
        self.model: Any = model
        self.processor, self.device = processor, device
        self.n_states = len(self.model.encoder.layers) + 1

    def encode(self, frames: list[Image.Image]) -> EncoderOutput:
        """Run under torch.inference_mode(). Inputs are moved to device and cast to the model dtype."""
        inputs = _move_inputs(self.processor(images=frames, return_tensors="pt"), self.model, self.device)
        with torch.inference_mode(), StateTap(self.model.encoder.layers) as tap:
            output = self.model(inputs["pixel_values"])
            kinds = pool(tap.states(), prefix=0, cls=False)
            head = {"pooler_mean": output.pooler_output.float().mean(0, keepdim=True).cpu()}
        return EncoderOutput(kinds, head)


class Dinov3Adapter:
    """dinov3-vitl: DINOv3ViTModel, image input.
    model: DINOv3ViTModel; processor: DINOv3ViTImageProcessor.
    States: StateTap(model.model.layer); token 0 is CLS, then config.num_register_tokens register tokens;
    prefix = 1 + num_register_tokens. Kinds patch_mean, patch_max, cls_mean. No head."""

    name = "dinov3-vitl"
    kinds = ("patch_mean", "patch_max", "cls_mean")
    head_kinds = ()
    n_states: int

    def __init__(self, model: torch.nn.Module, processor: Any, device: str = "cpu") -> None:
        self.model: Any = model
        self.processor, self.device = processor, device
        self.n_states = len(self.model.model.layer) + 1

    def encode(self, frames: list[Image.Image]) -> EncoderOutput:
        """Run under torch.inference_mode(). Inputs are moved to device and cast to the model dtype."""
        inputs = _move_inputs(self.processor(images=frames, return_tensors="pt"), self.model, self.device)
        with torch.inference_mode(), StateTap(self.model.model.layer) as tap:
            self.model(inputs["pixel_values"])
            kinds = pool(tap.states(), prefix=1 + self.model.config.num_register_tokens, cls=True)
        return EncoderOutput(kinds, {})


class Vjepa2Adapter:
    """vjepa2-vitl: VJEPA2Model, 8-frame video clip input.
    model: VJEPA2Model; processor: VJEPA2VideoProcessor.
    The 8 frames are one clip: processor(videos=[frames])["pixel_values_videos"] has shape (1, 8, 3, crop, crop).
    Forward model(pixel_values_videos=..., skip_predictor=True). States: StateTap(model.encoder.layer); prefix 0.
    Kinds patch_mean, patch_max. No head."""

    name = "vjepa2-vitl"
    kinds = ("patch_mean", "patch_max")
    head_kinds = ()
    n_states: int

    def __init__(self, model: torch.nn.Module, processor: Any, device: str = "cpu") -> None:
        self.model: Any = model
        self.processor, self.device = processor, device
        self.n_states = len(self.model.encoder.layer) + 1

    def encode(self, frames: list[Image.Image]) -> EncoderOutput:
        """Run under torch.inference_mode(). Inputs are moved to device and cast to the model dtype."""
        inputs = _move_inputs(self.processor(videos=[frames], return_tensors="pt"), self.model, self.device)
        with torch.inference_mode(), StateTap(self.model.encoder.layer) as tap:
            self.model(pixel_values_videos=inputs["pixel_values_videos"], skip_predictor=True)
            kinds = pool(tap.states(), prefix=0, cls=False)
        return EncoderOutput(kinds, {})


def load_encoder(name: str, asset_path: Path, device: str) -> Any:
    """Build the adapter for name from the asset folder: from_pretrained of the model (bfloat16 on cuda, float32 on
    cpu) and its processor, model.eval(). The only function in this module that calls from_pretrained.
    qwen35-vit: Qwen3_5VisionModel from the vision_config of asset_path/config.json, with the "model.visual." tensors
    of the safetensors shards. internvit-300m: InternVLModel, then the language model is deleted.
    Raise ValueError naming the encoder when name is not in ENCODERS."""
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder: {name}")

    from safetensors import safe_open
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoVideoProcessor,
        DINOv3ViTModel,
        InternVLModel,
        Qwen3_5VisionModel,
        SiglipImageProcessor,
        SiglipVisionModel,
        VJEPA2Model,
    )

    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    adapter: Any
    if name == "qwen35-vit":
        config = AutoConfig.from_pretrained(asset_path).vision_config
        model = Qwen3_5VisionModel._from_config(config, torch_dtype=dtype)
        try:
            with (asset_path / "model.safetensors.index.json").open() as file:
                weight_map = json.load(file)["weight_map"]
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid qwen35-vit weight index in {asset_path}") from exc
        weights = {}
        for shard in {shard for key, shard in weight_map.items() if key.startswith("model.visual.")}:
            with safe_open(asset_path / shard, framework="pt", device="cpu") as file:
                for key, filename in weight_map.items():
                    if filename == shard and key.startswith("model.visual."):
                        weights[key.removeprefix("model.visual.")] = file.get_tensor(key)
        model.load_state_dict(weights, strict=True)
        processor = AutoImageProcessor.from_pretrained(asset_path)
        adapter = QwenVitAdapter
    elif name == "internvit-300m":
        model = InternVLModel.from_pretrained(asset_path, torch_dtype=dtype)
        model.language_model = None
        processor = AutoImageProcessor.from_pretrained(asset_path, crop_to_patches=False)
        adapter = InternVitAdapter
    elif name == "siglip2-so400m":
        model = SiglipVisionModel.from_pretrained(asset_path, torch_dtype=dtype)
        processor = SiglipImageProcessor.from_pretrained(asset_path)
        adapter = SiglipAdapter
    elif name == "dinov3-vitl":
        model = DINOv3ViTModel.from_pretrained(asset_path, torch_dtype=dtype)
        processor = AutoImageProcessor.from_pretrained(asset_path)
        adapter = Dinov3Adapter
    else:
        model = VJEPA2Model.from_pretrained(asset_path, torch_dtype=dtype)
        processor = AutoVideoProcessor.from_pretrained(asset_path)
        adapter = Vjepa2Adapter
    return cast(Any, adapter)(cast(torch.nn.Module, model).to(device).eval(), processor, device)
