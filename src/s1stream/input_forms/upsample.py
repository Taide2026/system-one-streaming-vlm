"""Input form "upsample" (2026-11-11): every frame of the window enlarged by one factor with one kernel.

The whole-frame counterpart of "foveated" (which mixes 4 original and 4 enlarged frames): group A of the upsampling
study and the pure-2x training input. cv2 kernels, so bicubic matches the foveated form (cv2.INTER_CUBIC).
Config: {name: upsample, factor: 2.0, kernel: bicubic | nearest | bilinear | lanczos}.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from s1stream.input_forms.frames import read_frames
from s1stream.registry import INPUT_FORMS
from s1stream.types import ModelInput, Question, VideoItem, Window

KERNELS = {"bicubic": cv2.INTER_CUBIC, "nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR,
           "lanczos": cv2.INTER_LANCZOS4, "area": cv2.INTER_AREA}


def upsample(images: list, factor: float, kernel: str = "bicubic") -> list:
    if factor == 1:
        return images
    out = []
    for im in images:
        a = np.asarray(im.convert("RGB"))
        h, w = a.shape[:2]
        out.append(Image.fromarray(cv2.resize(a, (int(w * factor), int(h * factor)), interpolation=KERNELS[kernel])))
    return out


@INPUT_FORMS.register("upsample")
class Upsample:
    def __init__(self, factor: float = 2.0, kernel: str = "bicubic") -> None:
        self.factor, self.kernel = float(factor), str(kernel)
        if self.kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel}; one of {sorted(KERNELS)}")

    def build(self, item: VideoItem, window: Window, question: Question) -> ModelInput:
        images = read_frames(item.path, window.frame_indices)
        return ModelInput(images=upsample(images, self.factor, self.kernel), text=None)
