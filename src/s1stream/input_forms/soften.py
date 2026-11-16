"""Input form "soften" (2026-11-16): the frame at its own size, with less detail.

Every frame is area-downscaled to height `via` and enlarged back to its original size with one kernel, so the token
count does not change and only the sharpness does. On the 360p variant, via=180 is (nearly) the "180p enlarged 2x"
input and via=360 the real 360p frame; 240 and 300 lie between them.
Config: {name: soften, via: 240, kernel: bicubic}.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from s1stream.input_forms.frames import read_frames
from s1stream.input_forms.upsample import KERNELS
from s1stream.registry import INPUT_FORMS
from s1stream.types import ModelInput, Question, VideoItem, Window


def soften(images: list, via: int, kernel: str = "bicubic") -> list:
    out = []
    for im in images:
        a = np.asarray(im.convert("RGB"))
        h, w = a.shape[:2]
        if via >= h:
            out.append(im.convert("RGB")); continue
        small = cv2.resize(a, (int(round(w * via / h)), int(via)), interpolation=cv2.INTER_AREA)
        out.append(Image.fromarray(cv2.resize(small, (w, h), interpolation=KERNELS[kernel])))
    return out


@INPUT_FORMS.register("soften")
class Soften:
    def __init__(self, via: int = 240, kernel: str = "bicubic") -> None:
        self.via, self.kernel = int(via), str(kernel)
        if self.kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel}; one of {sorted(KERNELS)}")

    def build(self, item: VideoItem, window: Window, question: Question) -> ModelInput:
        return ModelInput(images=soften(read_frames(item.path, window.frame_indices), self.via, self.kernel), text=None)
