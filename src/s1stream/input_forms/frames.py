"""Input form "frames" (V1): the window's frames as RGB images."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
from PIL import Image

from s1stream.registry import INPUT_FORMS
from s1stream.types import ModelInput, Question, VideoItem, Window


class FrameReadError(Exception):
    """A frame could not be read. The message names the video and the frame index."""


def read_frames(path: Path, indices: Sequence[int]) -> list:
    """Return one PIL.Image (mode RGB) per index, in order. Raise FrameReadError when a frame cannot be read."""
    cap = cv2.VideoCapture(str(path))
    images = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise FrameReadError(f"Could not read {path} at frame {index}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
    finally:
        cap.release()
    return images


@INPUT_FORMS.register("frames")
class Frames:
    def build(self, item: VideoItem, window: Window, question: Question) -> ModelInput:
        """ModelInput(images=read_frames(item.path, window.frame_indices), text=None)."""
        return ModelInput(images=read_frames(item.path, window.frame_indices), text=None)
