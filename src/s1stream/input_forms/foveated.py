"""Input form "foveated" (2026-10-31): 4 full frames + 4 fovea crops at the TGM motion peak, one forward.

TGM (temporal gradient magnitude) = the summed absolute difference of the window's 8 frames (blurred); its peak
positions a frac-of-frame box (the recipe of the zoom pilot, scripts/zoom-full.py: 1/3 box, 2x cubic up-scale).
Frames 0, 2, 4, 6 stay full view (periphery); frames 1, 3, 5, 7 become the crop (fovea), so the sequence still
spans the whole window at both scales and the frame count - and the RTF - is unchanged. Deck pp. 57-58.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from s1stream.input_forms.frames import read_frames
from s1stream.registry import INPUT_FORMS
from s1stream.types import ModelInput, Question, VideoItem, Window


def tgm_box(arrays: list[np.ndarray], frac: float) -> tuple[int, int, int, int]:
    """(y0, x0, h, w) of the frac-of-frame box centred on the TGM peak, clamped to the frame."""
    grey = [cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32) for a in arrays]
    heat = cv2.GaussianBlur(np.sum([np.abs(a - b) for a, b in zip(grey[1:], grey[:-1])], axis=0), (31, 31), 0)
    height, width = heat.shape
    h, w = int(height * frac), int(width * frac)
    y, x = np.unravel_index(int(heat.argmax()), heat.shape)
    return min(max(y - h // 2, 0), height - h), min(max(x - w // 2, 0), width - w), h, w


def foveate(images: list, frac: float = 1 / 3, upscale: int = 2) -> list:
    """The foveated window: even frames full view, odd frames the TGM crop (up-scaled). Identity below 4 frames."""
    if len(images) < 4:
        return images
    arrays = [np.asarray(im.convert("RGB")) for im in images]
    y0, x0, h, w = tgm_box(arrays, frac)
    out = []
    for k, (im, a) in enumerate(zip(images, arrays)):
        if k % 2 == 0:
            out.append(im)
        else:
            crop = cv2.resize(a[y0:y0 + h, x0:x0 + w], (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
            out.append(Image.fromarray(crop))
    return out


@INPUT_FORMS.register("foveated")
class Foveated:
    """Config: {name: foveated, frac: 0.333, upscale: 2}."""

    def __init__(self, frac: float = 1 / 3, upscale: int = 2) -> None:
        self.frac, self.upscale = float(frac), int(upscale)

    def build(self, item: VideoItem, window: Window, question: Question) -> ModelInput:
        images = read_frames(item.path, window.frame_indices)
        return ModelInput(images=foveate(images, self.frac, self.upscale), text=None)
