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


def foveate(images: list, frac: float = 1 / 3, upscale: int = 2, max_area: float | None = None,
            mode: str = "tgm") -> list:
    """The foveated window: even frames full view, odd frames the TGM crop (up-scaled). Identity below 4 frames.
    max_area: skip foveation (return the frames unchanged) when the TGM hot area (> half peak) already covers at
    least this fraction of the frame - a large event fills the view, a crop only wastes frames (jev-hard large
    tier: always-foveate cost 10 AP, 2026-11-01)."""
    if len(images) < 4:
        return images
    arrays = [np.asarray(im.convert("RGB")) for im in images]
    if max_area is not None:
        grey = [cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32) for a in arrays]
        heat = cv2.GaussianBlur(np.sum([np.abs(x - y) for x, y in zip(grey[1:], grey[:-1])], axis=0), (31, 31), 0)
        peak = float(heat.max())
        if peak > 0 and float((heat > 0.5 * peak).mean()) >= max_area:
            return images
    if mode == "center":  # WHERE-ablation control: the same 4+4 format with the crop fixed at the frame centre
        height, width = arrays[0].shape[:2]
        h, w = int(height * frac), int(width * frac)
        y0, x0 = (height - h) // 2, (width - w) // 2
    else:
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
    """Config: {name: foveated, frac: 0.333, upscale: 2, max_area: null | 0.045}."""

    def __init__(self, frac: float = 1 / 3, upscale: int = 2, max_area: float | None = None,
                 mode: str = "tgm") -> None:
        self.frac, self.upscale, self.mode = float(frac), int(upscale), str(mode)
        self.max_area = None if max_area is None else float(max_area)

    def build(self, item: VideoItem, window: Window, question: Question) -> ModelInput:
        images = read_frames(item.path, window.frame_indices)
        return ModelInput(images=foveate(images, self.frac, self.upscale, self.max_area, self.mode), text=None)


@INPUT_FORMS.register("foveated_gated")
class FoveatedGated:
    """Active foveation, the WHEN gate driven by the model itself (2026-11-02): window t+1 is foveated iff the
    model's own p at window t fell in the unsure band (band[0] <= p <= band[1]) - "the second look after a first
    suspicion". The first window of every video is always the full view (causal; no signal yet). max_area still
    vetoes the crop when the event already fills the frame. Needs the stateful runner hooks (start_video /
    observe); prefetch is disabled for stateful forms. Config: {name: foveated_gated, band: [0.3, 0.7],
    frac: 0.333, upscale: 2, max_area: 0.045}."""

    def __init__(self, band: tuple[float, float] = (0.3, 0.7), frac: float = 1 / 3, upscale: int = 2,
                 max_area: float | None = 0.045) -> None:
        self.band = (float(band[0]), float(band[1]))
        self.frac, self.upscale = float(frac), int(upscale)
        self.max_area = None if max_area is None else float(max_area)
        self.last_p: float | None = None

    def start_video(self) -> None:
        self.last_p = None

    def observe(self, score: float) -> None:
        self.last_p = float(score)

    def build(self, item: VideoItem, window: Window, question: Question) -> ModelInput:
        images = read_frames(item.path, window.frame_indices)
        if self.last_p is not None and self.band[0] <= self.last_p <= self.band[1]:
            images = foveate(images, self.frac, self.upscale, self.max_area)
        return ModelInput(images=images, text=None)
