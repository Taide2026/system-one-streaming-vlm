"""Backbone "startlux": StartLux-Decision v1.1 Instant (startlux-models) through the release's own package,
startlux-decide (github.com/StartLuxLabs/StartLux-Decision @ 83fe6c7, Apache-2.0), in process on its transformers
engine, which reads an Instant folder with the bidirectional prompt attention it was trained with.

One window is one System One request, as in the d3 backbone:
  Decider.decide(state=question.state, images=<frames>,
                 questions={"answer": {"type": "choice", "instructions": question.text,
                                       "criteria": {option: None, ...}}})
with the folder's default controls: no thought (Instant), none_check off, so the read-out covers the listed options
only (the package's server turns none_check on by default). forward returns the logs of the option probabilities in
question.options order (option_logits).

The frames go in as PIL images: FrameDecider builds the package's Media from them as images.load_media does from a
request's base64 strings (the same resize bounds and token counts), without the encode / decode round trip.
"""
from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

import torch

from s1stream.backbones.d3 import QUESTION_ID, build_questions
from s1stream.registry import BACKBONES
from s1stream.types import BackboneOutput, ModelInput, Question


def allow_transformers_5_17() -> None:
    """The package's bidirectional.check_body refuses every transformers but 5.8.x; this repository pins 5.17.0. Swap
    in the same checks without that version test. bidirectional.enable still checks on the model itself that its
    decoder loop equals the stock forward, that padding reaches no token and that a late token reaches the first; the
    image path (features, multimodal positions) was checked once against the stock causal forward on 5.17.0. 5.17 has
    no _update_linear_attn_mask (text_forward calls it): check_body adds its 5.17 equivalent to the decoder."""
    from startlux_decide.backends import bidirectional

    if getattr(bidirectional.check_body, "s1stream", False):
        return

    def linear_attn_mask(attention_mask: Any, cache: Any) -> Any:
        # 5.17 builds this mask with masking_utils.create_recurrent_attention_mask: None without padding, else the
        # 2D mask (no cache here, so it already has the sequence's length).
        if attention_mask is None or attention_mask.ndim != 2 or bool((attention_mask == 1).all()):
            return None
        return attention_mask

    def check_body(body: Any) -> dict:
        cfg = body.config
        if not hasattr(body, "_update_linear_attn_mask"):      # 5.8's method, used by bidirectional.text_forward
            body._update_linear_attn_mask = linear_attn_mask
        if cfg._attn_implementation != "sdpa":
            raise ValueError(f"bidirectional prompt attention needs sdpa attention, not {cfg._attn_implementation!r}")
        for name in ("embed_tokens", "rotary_emb", "layers", "norm", "_update_linear_attn_mask"):
            if not hasattr(body, name):
                raise ValueError(f"the text decoder lacks {name}: not the hybrid decoder of these models")
        types = list(cfg.layer_types[:cfg.num_hidden_layers])
        if "full_attention" not in types or set(types) - {"full_attention", "linear_attention"}:
            raise ValueError(f"unexpected layer types {sorted(set(types))}")
        for i, layer in enumerate(body.layers[:cfg.num_hidden_layers]):
            # 5.17 layers carry no layer_type: the config's type and the attention module must agree.
            if (types[i] == "full_attention") != hasattr(layer, "self_attn"):
                raise ValueError(f"layer {i} does not match its declared type {types[i]}")
        return {"full_attention_layers": [i for i, t in enumerate(types) if t == "full_attention"],
                "layers": len(types)}

    check_body.s1stream = True
    bidirectional.check_body = check_body


def frame_decider(folder: Any, backend: Any) -> Any:
    """A startlux_decide Decider whose request images are PIL images (the frames) instead of base64 strings."""
    from startlux_decide.images import Media, place_images, smart_resize
    from startlux_decide.pipeline import Decider
    from startlux_decide.prompt import state_text

    class FrameDecider(Decider):
        def _images(self, state: Any, images: Any) -> tuple[str, Any]:
            if len(images) > self.max_images:
                raise ValueError(f"at most {self.max_images} images per request, not {len(images)}")
            factor = self.folder.image_factor
            media = Media(min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            for image in images:
                image = image.convert("RGB")
                height, width = smart_resize(image.height, image.width, factor, self.min_pixels, self.max_pixels)
                media.raw.append((b"", ""))
                media.images.append(image)
                media.sizes.append((height, width))
                media.tokens.append((height // factor) * (width // factor))
            return place_images(state_text(state), len(media)), media

    return FrameDecider(folder, backend)


@BACKBONES.register("startlux")
class StartLuxBackbone:
    """Config: {name: startlux, asset: startlux-4b-instant | startlux-9b-instant, memory_cap_gb: 40}.
    load_config adds path."""

    def __init__(self, path: str | Path, memory_cap_gb: float = 40.0, device: str = "cuda",
                 asset: str | None = None) -> None:
        """On cuda, cap the GPU memory as jev_omni does; load the folder (Instant only) on the package's transformers
        engine in bfloat16; it reads the files of path only."""
        from startlux_decide.backends.transformers import TransformersBackend
        from startlux_decide.modelcfg import load_model_folder, load_tokenizer

        if device.startswith("cuda"):
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, memory_cap_gb * 2**30 / total_memory))
        folder = load_model_folder(str(path))
        if folder.kind != "instant":
            raise ValueError(f"{path} is a {folder.kind} folder; the startlux backbone reads Instant folders only")
        allow_transformers_5_17()
        backend = TransformersBackend(str(path), load_tokenizer(str(path)), folder.bidirectional, device, "bfloat16")
        loop = asyncio.new_event_loop()
        loop.run_until_complete(backend.start())
        if not backend.reads_images:
            raise ValueError(f"{path} has no image processor: the startlux backbone needs the frames")
        self._init(frame_decider(folder, backend), loop)

    def _init(self, decider: Any, loop: asyncio.AbstractEventLoop) -> None:
        self.decider, self.loop = decider, loop

    @classmethod
    def from_parts(cls, decider: Any, loop: asyncio.AbstractEventLoop | None = None) -> "StartLuxBackbone":
        """Build without loading files (used by tests and by __init__)."""
        instance = cls.__new__(cls)
        instance._init(decider, loop or asyncio.new_event_loop())
        return instance

    def forward(self, inp: ModelInput, question: Question) -> BackboneOutput:
        """One decide call with the frames as images.

        Return BackboneOutput(logits=<float32 CPU tensor, log p(option) in question.options order>).
        Raise RuntimeError with the package's message when it answers with an error.
        """
        decision = self.loop.run_until_complete(
            self.decider.decide(question.state, build_questions(question), images=list(inp.images)))
        answer = decision.answers[QUESTION_ID]
        if "error" in answer:
            raise RuntimeError(f"StartLux could not answer: {answer['error']}: {answer.get('message', '')}")
        probabilities = answer["probabilities"]
        return BackboneOutput(logits=torch.tensor(
            [math.log(max(probabilities[option], 1e-30)) for option in question.options], dtype=torch.float32))
