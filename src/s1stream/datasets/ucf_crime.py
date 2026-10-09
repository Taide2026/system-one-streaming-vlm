"""Dataset "ucf_crime": UCF-Crime test videos with temporal annotations.

Asset layout (see the asset README): <root>/videos/<Class>/<name>, <root>/videos/Testing_Normal_Videos_Anomaly/<name>,
<root>/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt.
Annotation line: "Arrest001_x264.mp4  Arrest  1185  1485  -1  -1" (name, category, two start/end frame pairs; -1 = none).
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2

from s1stream.registry import DATASETS
from s1stream.types import VideoItem

ANNOTATION_FILE = "annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
NORMAL_CATEGORY = "Normal"
NORMAL_TEST_FOLDER = "Testing_Normal_Videos_Anomaly"

Probe = Callable[[Path], tuple[float, int]]  # path -> (fps, frame count)


class DatasetError(Exception):
    """A dataset problem. The message names the file or the video."""


@dataclass(frozen=True)
class AnnotationRow:
    name: str
    category: str
    events: tuple[tuple[int, int], ...]


def parse_annotation_line(line: str) -> AnnotationRow:
    """Parse one annotation line. Drop a (start, end) pair when start is -1. Raise DatasetError on a malformed line."""
    fields = line.split()
    if len(fields) < 2 or (len(fields) - 2) % 2:
        raise DatasetError(f"Malformed annotation line: {line!r}")
    try:
        pairs = [(int(fields[i]), int(fields[i + 1])) for i in range(2, len(fields), 2)]
    except ValueError as exc:
        raise DatasetError(f"Malformed annotation line: {line!r}") from exc
    events = tuple((start, end) for start, end in pairs if start != -1)
    return AnnotationRow(fields[0], fields[1], events)


def read_annotations(path: Path) -> list[AnnotationRow]:
    """Parse every non-empty line of the annotation file, in file order."""
    with path.open() as annotations:
        return [parse_annotation_line(line) for line in annotations if line.strip()]


def video_path(root: Path, row: AnnotationRow) -> Path:
    """Return root/videos/<category>/<name>, or root/videos/Testing_Normal_Videos_Anomaly/<name> for Normal.

    Raise DatasetError that names the video when the file does not exist.
    """
    folder = NORMAL_TEST_FOLDER if row.category == NORMAL_CATEGORY else row.category
    path = root / "videos" / folder / row.name
    if not path.is_file():
        raise DatasetError(f"Video not found: {row.name} ({path})")
    return path


def probe_video(path: Path) -> tuple[float, int]:
    """Read fps and frame count with OpenCV. Raise DatasetError when the file cannot be opened."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise DatasetError(f"Cannot open video: {path}")
        return float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def select(rows: Sequence[AnnotationRow], root: Path, probe: Probe, n_anomaly: int, n_normal: int,
           max_duration_s: float, seed: int) -> list[VideoItem]:
    """Pick n_anomaly anomaly rows and n_normal Normal rows whose duration is at most max_duration_s.

    Algorithm (fixed, so a seed always gives the same list):
    1. Split rows into anomaly and Normal; sort each group by name.
    2. For each group, shuffle with random.Random(seed) (one new Random per group), then walk the list:
       probe the video, keep it when n_frames / fps <= max_duration_s, stop at n kept.
    3. Raise DatasetError naming the group when fewer than n videos fit.
    Return anomaly items then Normal items, each group sorted by name.
    """
    selected: list[VideoItem] = []
    for category, needed in (("anomaly", n_anomaly), ("Normal", n_normal)):
        group = [row for row in rows if (row.category == NORMAL_CATEGORY) == (category == "Normal")]
        group.sort(key=lambda row: row.name)
        random.Random(seed).shuffle(group)
        kept: list[VideoItem] = []
        for row in group:
            path = video_path(root, row)
            fps, n_frames = probe(path)
            if fps <= 0:
                continue
            if n_frames / fps <= max_duration_s:
                kept.append(VideoItem(row.name, path, row.category, fps, n_frames, row.events))
            if len(kept) == needed:
                break
        if len(kept) < needed:
            raise DatasetError(f"Too few {category} videos fit the duration limit")
        selected.extend(sorted(kept, key=lambda item: item.name))
    return selected


@DATASETS.register("ucf_crime")
class UcfCrime:
    """Config: {name: ucf_crime, asset: ucf-crime, n_anomaly, n_normal, max_duration_s, seed}. load_config adds path."""

    def __init__(self, path: str | Path, n_anomaly: int, n_normal: int, max_duration_s: float, seed: int,
                 asset: str | None = None, probe: Probe = probe_video) -> None:
        self.root = Path(path)
        self.n_anomaly, self.n_normal = n_anomaly, n_normal
        self.max_duration_s, self.seed = max_duration_s, seed
        self.probe = probe

    def items(self) -> list[VideoItem]:
        """read_annotations(root / ANNOTATION_FILE), then select(...)."""
        rows = read_annotations(self.root / ANNOTATION_FILE)
        return select(rows, self.root, self.probe, self.n_anomaly, self.n_normal,
                      self.max_duration_s, self.seed)
