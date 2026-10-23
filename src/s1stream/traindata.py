"""Training records for the read-out-level fine-tune (stage 1, 2026-10-24): windows of the UCF-Crime training videos.

The UCF-Crime training split carries one label per video. HIVAU-70k (Holmes-VAU, CVPR 2025;
github.com/pipixin321/HolmesVAU, HIVAU-70k/raw_annotations/ucf_database_train.json) adds the start and end of every
anomaly event, in seconds:
  {"Abuse001_x264": {"n_frames": 2729, "fps": 30.0, "label": ["Abuse"], "events": [[3.933, 15.867], ...], ...}, ...}
Normal training videos are in there too, with "events": [].

A record is one window of a training video, cut exactly as at inference (s1stream.windows.plan_windows: length,
stride, frames), with a soft label y = coverage = the share of the window's time inside an event (0 outside, 1 inside,
a fraction at the edges). The loss (s1stream.train) is the soft cross-entropy over the question's options with
(positive: y, the other option: 1 - y).

Sampling (sample_records): the videos are split by a seed into training videos and held-out videos (the temperature
is fitted on the held-out ones, never on test videos); from the training videos n_abnormal windows with y > 0 and
n_normal windows with y == 0 (from normal videos and the normal stretches of anomaly videos) are drawn without
replacement; the held-out videos give n_val windows, half abnormal. Records are written to records.jsonl so a run is
reproducible from its folder.
"""
from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from s1stream.types import VideoItem, Window
from s1stream.windows import plan_windows

NORMAL_TRAIN_FOLDER = "Training_Normal_Videos_Anomaly"


class TrainDataError(Exception):
    """A training-data problem. The message names the file or the video."""


@dataclass(frozen=True)
class Record:
    """One training window. frame_indices are the frames read from path; y is the coverage in [0, 1]."""
    video: str
    path: str
    category: str
    window: int
    start_s: float
    end_s: float
    frame_indices: tuple[int, ...]
    y: float


def video_folder(name: str) -> str:
    """The UCF-Crime folder of a training video from its name: Normal_Videos... -> NORMAL_TRAIN_FOLDER, else the
    class prefix before the digits (RoadAccidents021_x264 -> RoadAccidents)."""
    if name.startswith("Normal_Videos"):
        return NORMAL_TRAIN_FOLDER
    prefix = name.rstrip("_x264")
    while prefix and prefix[-1].isdigit():
        prefix = prefix[:-1]
    return prefix


def read_hivau(json_path: Path, root: Path) -> list[VideoItem]:
    """The HIVAU-70k training videos as VideoItems (events in inclusive frames, rounded from seconds), sorted by
    name. Videos whose file is missing under root/videos/<folder>/<name>.mp4 raise TrainDataError naming the first."""
    try:
        data = json.loads(json_path.read_text())
    except (OSError, ValueError) as exc:
        raise TrainDataError(f"Could not read {json_path}: {exc}") from exc
    items: list[VideoItem] = []
    for name in sorted(data):
        entry = data[name]
        fps, n_frames = float(entry["fps"]), int(entry["n_frames"])
        if fps <= 0 or n_frames <= 0:
            raise TrainDataError(f"{name}: fps {fps}, n_frames {n_frames}")
        path = root / "videos" / video_folder(name) / f"{name}.mp4"
        if not path.is_file():
            raise TrainDataError(f"Video not found: {path}")
        events = tuple((round(start * fps), max(round(start * fps), round(end * fps) - 1))
                       for start, end in entry.get("events", []))
        category = "Normal" if not events else video_folder(name)
        items.append(VideoItem(name, path, category, fps, n_frames, events))
    return items


def coverage(window: Window, item: VideoItem) -> float:
    """The share of the window's time [start_s, end_s) inside the item's events (seconds, from frames at fps)."""
    length = window.end_s - window.start_s
    if length <= 0:
        return 0.0
    covered = 0.0
    for start_frame, end_frame in item.events:
        start, end = start_frame / item.fps, (end_frame + 1) / item.fps
        covered += max(0.0, min(end, window.end_s) - max(start, window.start_s))
    return min(1.0, covered / length)


def windows_of(item: VideoItem, length_s: float, stride_s: float, frames: int) -> list[Record]:
    """Every window of item as a Record with its coverage."""
    return [Record(item.name, str(item.path), item.category, window.index, window.start_s, window.end_s,
                   window.frame_indices, coverage(window, item))
            for window in plan_windows(item, length_s, stride_s, frames)]


def split_videos(items: Sequence[VideoItem], holdout: float, seed: int) -> tuple[list[VideoItem], list[VideoItem]]:
    """(training videos, held-out videos): a seeded shuffle, holdout share held out, per category group (anomaly /
    Normal) so both sides hold both kinds."""
    train: list[VideoItem] = []
    held: list[VideoItem] = []
    for normal in (False, True):
        group = sorted((item for item in items if (item.category == "Normal") == normal), key=lambda i: i.name)
        random.Random(seed + int(normal)).shuffle(group)
        n_held = int(round(len(group) * holdout))
        held.extend(group[:n_held])
        train.extend(group[n_held:])
    return sorted(train, key=lambda i: i.name), sorted(held, key=lambda i: i.name)


def draw(records: Sequence[Record], n_abnormal: int, n_normal: int, seed: int) -> list[Record]:
    """n_abnormal records with y > 0 and n_normal with y == 0, drawn without replacement with the seed, shuffled
    together. Raise TrainDataError when a pool is too small."""
    abnormal = [record for record in records if record.y > 0]
    normal = [record for record in records if record.y == 0]
    if len(abnormal) < n_abnormal or len(normal) < n_normal:
        raise TrainDataError(f"Too few windows: {len(abnormal)} abnormal (need {n_abnormal}), "
                             f"{len(normal)} normal (need {n_normal})")
    rng = random.Random(seed)
    chosen = rng.sample(abnormal, n_abnormal) + rng.sample(normal, n_normal)
    rng.shuffle(chosen)
    return chosen


def sample_records(items: Sequence[VideoItem], length_s: float, stride_s: float, frames: int, n_abnormal: int,
                   n_normal: int, n_val: int, holdout: float, seed: int) -> tuple[list[Record], list[Record]]:
    """(training records, held-out records); see the module docstring."""
    train_videos, held_videos = split_videos(items, holdout, seed)
    train_pool = [record for item in train_videos for record in windows_of(item, length_s, stride_s, frames)]
    held_pool = [record for item in held_videos for record in windows_of(item, length_s, stride_s, frames)]
    train = draw(train_pool, n_abnormal, n_normal, seed)
    val = draw(held_pool, n_val // 2, n_val - n_val // 2, seed + 1) if n_val else []
    return train, val


def write_records(records: Sequence[Record], path: Path) -> None:
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record)) + "\n")


def read_records(path: Path) -> list[Record]:
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            data = json.loads(line)
            data["frame_indices"] = tuple(data["frame_indices"])
            records.append(Record(**data))
    return records
