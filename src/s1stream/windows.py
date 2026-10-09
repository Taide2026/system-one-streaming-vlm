"""The windower: cut a video into sliding windows. Not a slot; every experiment uses it."""
from __future__ import annotations

from s1stream.types import VideoItem, Window


def window_frame_range(start_s: float, end_s: float, fps: float) -> tuple[int, int]:
    """Inclusive frame range of a window: (round(start_s * fps), round(end_s * fps) - 1)."""
    raise NotImplementedError


def plan_windows(item: VideoItem, length_s: float, stride_s: float, frames: int) -> list[Window]:
    """Windows start at 0, stride_s, 2 * stride_s, ...; a window is kept only when start + length_s <= duration + 1e-9.

    frame_indices[k] = round((start + (k + 0.5) * length_s / frames) * fps) for k in 0 .. frames - 1.
    label = 1 when window_frame_range overlaps any event range of item (inclusive), else 0.
    """
    raise NotImplementedError
