"""Input form "keycrop" (2026-11-13): key frames + motion crops as plain images, for any model that encodes each
image at its own size. The model-agnostic form of scripts/keycrop.py: nothing is placed at token positions; the
model simply receives, in temporal order, the key frames whole and the other frames as a crop.

  key frames    enlarged by `factor` (bicubic) and sent whole
  other frames  the bounding box of the moving blocks (blocks of `block` pixels that the motion region covers by more
                than `thr`), cut from the original frame and enlarged by `factor`; when nothing moves, a frame
                contributes nothing
Motion region: pixels above half the peak of the summed, blurred absolute frame difference of the window (the same
map as scripts/mech.py tgm_mask).
Config: {name: keycrop, key: [0, 7], factor: 2.0, block: 32, thr: 0.05}.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from s1stream.input_forms.frames import read_frames
from s1stream.registry import INPUT_FORMS
from s1stream.types import ModelInput, Question, VideoItem, Window


def motion_mask(arrays: list) -> np.ndarray:
    grey = [cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32) for a in arrays]
    heat = cv2.GaussianBlur(np.sum([np.abs(x - y) for x, y in zip(grey[1:], grey[:-1])], axis=0), (31, 31), 0)
    peak = float(heat.max())
    return (heat > 0.5 * peak).astype(np.float32) if peak > 0 else np.zeros_like(heat)


def keycrop_images(images: list, key: tuple[int, ...], factor: float, block: int, thr: float) -> list:
    arrays = [np.asarray(im.convert("RGB")) for im in images]
    height, width = arrays[0].shape[:2]
    hb, wb = -(-height // block), -(-width // block)
    fine = cv2.resize(motion_mask(arrays), (wb, hb), interpolation=cv2.INTER_AREA) > thr
    box = None
    if fine.any():
        rr, cc = np.where(fine)
        box = (cc.min() * block, min((cc.max() + 1) * block, width), rr.min() * block, min((rr.max() + 1) * block, height))
    big = lambda a: Image.fromarray(cv2.resize(a, (round(a.shape[1] * factor), round(a.shape[0] * factor)), interpolation=cv2.INTER_CUBIC))
    out = []
    for k, a in enumerate(arrays):
        if k in key:
            out.append(big(a))
        elif box is not None:
            out.append(big(a[box[2]:box[3], box[0]:box[1]]))
    return out


@INPUT_FORMS.register("keycrop")
class KeyCrop:
    def __init__(self, key: list[int] | None = None, factor: float = 2.0, block: int = 32, thr: float = 0.05) -> None:
        self.key, self.factor, self.block, self.thr = tuple(key if key is not None else (0, 7)), float(factor), int(block), float(thr)

    def build(self, item: VideoItem, window: Window, question: Question) -> ModelInput:
        images = read_frames(item.path, window.frame_indices)
        return ModelInput(images=keycrop_images(images, self.key, self.factor, self.block, self.thr), text=None)
