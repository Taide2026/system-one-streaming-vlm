"""Data types shared by every slot. They hold data only, no logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoItem:
    """One video of a dataset.

    events: inclusive (start_frame, end_frame) ranges of the annotated events. Empty for a normal video.
    """
    name: str
    path: Path
    category: str
    fps: float
    n_frames: int
    events: tuple[tuple[int, int], ...] = ()

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps


@dataclass(frozen=True)
class Window:
    """One sliding window of a video. The decision time of a window is end_s.

    frame_indices: the frames given to the model, increasing.
    label: 1 when the window overlaps an event by at least 1 frame, else 0.
    """
    video: str
    index: int
    start_s: float
    end_s: float
    frame_indices: tuple[int, ...]
    label: int


@dataclass(frozen=True)
class Question:
    """A typed question. positive is the option whose probability is the abnormal score."""
    state: str
    text: str
    options: tuple[str, ...]
    positive: str


@dataclass
class ModelInput:
    """What V1 (input form) gives V2 (backbone): images (PIL.Image.Image, RGB), text, or both."""
    images: list[Any] = field(default_factory=list)
    text: str | None = None


@dataclass
class BackboneOutput:
    """What V2 (backbone) gives V3 (answer head). Tensors are detached and on the CPU.

    hidden: last hidden state at the last prompt position, shape (1, hidden_size), float32.
    logits: next-token logits at the same position, or None when the backbone does not compute them.
    """
    hidden: Any = None
    logits: Any = None
