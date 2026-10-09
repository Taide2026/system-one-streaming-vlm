"""Input form "frames" (V1): the window's frames as RGB images."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from s1stream.registry import INPUT_FORMS
from s1stream.types import ModelInput, Question, VideoItem, Window


class FrameReadError(Exception):
    """A frame could not be read. The message names the video and the frame index."""


def read_frames(path: Path, indices: Sequence[int]) -> list:
    """Return one PIL.Image (mode RGB) per index, in order. Raise FrameReadError when a frame cannot be read."""
    raise NotImplementedError


@INPUT_FORMS.register("frames")
class Frames:
    def build(self, item: VideoItem, window: Window, question: Question) -> ModelInput:
        """ModelInput(images=read_frames(item.path, window.frame_indices), text=None)."""
        raise NotImplementedError
