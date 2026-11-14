"""Input form "diff" (2026-11-07, roadmap B2): 4 full frames + 4 temporal-difference frames, one forward.

Frames 0, 2, 4, 6 stay the full view; frames 1, 3, 5, 7 become gain * |frame_k - frame_{k-1}| (clipped to 255),
the cheapest explicit motion evidence a window can carry. Same frame count as "frames" and "foveated", so the RTF
is unchanged. Config: {name: diff, gain: 4}.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from s1stream.input_forms.frames import read_frames
from s1stream.registry import INPUT_FORMS
from s1stream.types import ModelInput, Question, VideoItem, Window


def temporal_diff(images: list, gain: float = 4.0) -> list:
    """Even frames unchanged; odd frame k -> gain * |frame_k - frame_{k-1}|. Identity below 4 frames."""
    if len(images) < 4:
        return images
    arrays = [np.asarray(im.convert("RGB")).astype(np.float32) for im in images]
    out = []
    for k, im in enumerate(images):
        if k % 2 == 0:
            out.append(im)
        else:
            d = np.clip(np.abs(arrays[k] - arrays[k - 1]) * float(gain), 0, 255).astype(np.uint8)
            out.append(Image.fromarray(d))
    return out


@INPUT_FORMS.register("diff")
class Diff:
    """Config: {name: diff, gain: 4}."""

    def __init__(self, gain: float = 4.0) -> None:
        self.gain = float(gain)

    def build(self, item: VideoItem, window: Window, question: Question) -> ModelInput:
        return ModelInput(images=temporal_diff(read_frames(item.path, window.frame_indices), self.gain), text=None)
