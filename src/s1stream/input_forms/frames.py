"""Input form "frames" (V1): the window's frames as RGB images."""
from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from s1stream.registry import INPUT_FORMS
from s1stream.types import ModelInput, Question, VideoItem, Window

# Skip up to this many frames with grab() before a seek is cheaper. Measured on UCF-Crime 320x240 H.264
# (2026-11-07): grab() 0.08-0.16 ms per frame, one seek + read 15-17 ms.
MAX_GRAB = 100


class FrameReadError(Exception):
    """A frame could not be read. The message names the video and the frame index."""


class SequentialReader:
    """One open video. Reads forward with grab() and seeks only on a backward jump or a gap above MAX_GRAB.

    The old read_frames opened the video and seeked once per frame: about 115 ms per 8-frame UCF window.
    Reading forward decodes the same frames without the seek cost.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        self.next = 0  # index of the frame that the next grab() decodes
        self.last: tuple[int, np.ndarray] | None = None  # (index, RGB) of the last frame read

    def close(self) -> None:
        self.cap.release()

    def seek(self, index: int) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.next = index

    def frame(self, index: int) -> np.ndarray:
        """The RGB array of frame `index`. Raise FrameReadError when it cannot be read."""
        if self.last is not None and self.last[0] == index:
            return self.last[1]
        if index < self.next or index - self.next > MAX_GRAB:
            self.seek(index)
        while self.next < index:
            if not self.cap.grab():
                raise FrameReadError(f"Could not read {self.path} at frame {index}")
            self.next += 1
        ok, frame = self.cap.read()
        if not ok:
            raise FrameReadError(f"Could not read {self.path} at frame {index}")
        self.next = index + 1
        self.last = (index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self.last[1]


# One reader per thread: runner prefetch threads and DataLoader workers each keep their own open video.
_local = threading.local()
_inherited: list[SequentialReader] = []  # readers a forked child got from its parent; kept, never used


def _forget_reader() -> None:
    """In a forked child (a DataLoader worker), set the parent's open video aside and open a new one on the next read.

    The child must not read or release the inherited VideoCapture: both hang, because FFmpeg's decoder threads do
    not exist in the child (checked 2026-11-07). The list keeps the object alive, so its destructor never runs.
    """
    reader = getattr(_local, "reader", None)
    if reader is not None:
        _inherited.append(reader)
    _local.reader = None


os.register_at_fork(after_in_child=_forget_reader)


def close_reader() -> None:
    """Release this thread's open video, if any."""
    reader = getattr(_local, "reader", None)
    if reader is not None:
        reader.close()
    _local.reader = None


def _reader(path: Path) -> SequentialReader:
    """This thread's reader for path; opens path when the thread last read another video."""
    reader = getattr(_local, "reader", None)
    if reader is not None and reader.path == path:
        return reader
    close_reader()
    _local.reader = SequentialReader(path)
    return _local.reader


def read_frames(path: Path, indices: Sequence[int]) -> list:
    """Return one PIL.Image (mode RGB) per index, in order. Raise FrameReadError when a frame cannot be read.

    The video stays open in this thread, so the next window of the same video continues where this one stopped.
    """
    reader = _reader(Path(path))
    try:
        return [Image.fromarray(reader.frame(int(index))) for index in indices]
    except FrameReadError:
        close_reader()  # the decoder state is unknown after a failed read; the next call reopens the video
        raise


@INPUT_FORMS.register("frames")
class Frames:
    def build(self, item: VideoItem, window: Window, question: Question) -> ModelInput:
        """ModelInput(images=read_frames(item.path, window.frame_indices), text=None)."""
        return ModelInput(images=read_frames(item.path, window.frame_indices), text=None)
