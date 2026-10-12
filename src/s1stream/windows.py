"""The windower: cut a video into sliding windows. Not a slot; every experiment uses it."""
from __future__ import annotations

from s1stream.types import VideoItem, Window


def window_frame_range(start_s: float, end_s: float, fps: float) -> tuple[int, int]:
    """Inclusive frame range of a window: (round(start_s * fps), round(end_s * fps) - 1)."""
    return round(start_s * fps), round(end_s * fps) - 1


def plan_windows(item: VideoItem, length_s: float, stride_s: float, frames: int) -> list[Window]:
    """Windows start at 0, stride_s, 2 * stride_s, ...; a window is kept only when start + length_s <= duration + 1e-9.

    frame_indices[k] = round((start + (k + 0.5) * length_s / frames) * fps) for k in 0 .. frames - 1.
    label = 1 when window_frame_range overlaps any event range of item (inclusive), else 0.
    """
    result: list[Window] = []
    duration = item.duration_s
    index = 0
    while True:
        start = index * stride_s
        end = start + length_s
        if end > duration + 1e-9:
            break

        # current frame range, e.g. frames on 30s - 27s
        frame_range = window_frame_range(start, end, item.fps)
        # calculation of current frames' distribution
        frame_indices = tuple(
            round((start + (k + 0.5) * length_s / frames) * item.fps)
            for k in range(frames)
        )
        label = 0
        for event_start, event_end in item.events:
            if frame_range[0] <= event_end and event_start <= frame_range[1]:
                label = 1
                break
        result.append(Window(item.name, index, start, end, frame_indices, label))
        index += 1
    return result
