"""Vision encoder adapters and loader (encoder-probing decisions 1, 3, 4)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch


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
        raise NotImplementedError

    def encode(self, frames: list[Image.Image]) -> EncoderOutput:
        """Run under torch.inference_mode(). Inputs are moved to device and cast to the model dtype."""
        raise NotImplementedError


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
        raise NotImplementedError

    def encode(self, frames: list[Image.Image]) -> EncoderOutput:
        """Run under torch.inference_mode(). Inputs are moved to device and cast to the model dtype."""
        raise NotImplementedError


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
        raise NotImplementedError

    def encode(self, frames: list[Image.Image]) -> EncoderOutput:
        """Run under torch.inference_mode(). Inputs are moved to device and cast to the model dtype."""
        raise NotImplementedError


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
        raise NotImplementedError

    def encode(self, frames: list[Image.Image]) -> EncoderOutput:
        """Run under torch.inference_mode(). Inputs are moved to device and cast to the model dtype."""
        raise NotImplementedError


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
        raise NotImplementedError

    def encode(self, frames: list[Image.Image]) -> EncoderOutput:
        """Run under torch.inference_mode(). Inputs are moved to device and cast to the model dtype."""
        raise NotImplementedError


def load_encoder(name: str, asset_path: Path, device: str) -> Any:
    """Build the adapter for name from the asset folder: from_pretrained of the model (bfloat16 on cuda, float32 on
    cpu) and its processor, model.eval(). The only function in this module that calls from_pretrained.
    qwen35-vit: Qwen3_5VisionModel from the vision_config of asset_path/config.json, with the "model.visual." tensors
    of the safetensors shards. internvit-300m: InternVLModel, then the language model is deleted.
    Raise ValueError naming the encoder when name is not in ENCODERS."""
    raise NotImplementedError
