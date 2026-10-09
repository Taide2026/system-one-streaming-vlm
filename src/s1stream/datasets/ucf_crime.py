"""Dataset "ucf_crime": UCF-Crime test videos with temporal annotations.

Asset layout (see the asset README): <root>/videos/<Class>/<name>, <root>/videos/Testing_Normal_Videos_Anomaly/<name>,
<root>/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt.
Annotation line: "Arrest001_x264.mp4  Arrest  1185  1485  -1  -1" (name, category, two start/end frame pairs; -1 = none).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

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
    raise NotImplementedError


def read_annotations(path: Path) -> list[AnnotationRow]:
    """Parse every non-empty line of the annotation file, in file order."""
    raise NotImplementedError


def video_path(root: Path, row: AnnotationRow) -> Path:
    """Return root/videos/<category>/<name>, or root/videos/Testing_Normal_Videos_Anomaly/<name> for Normal.

    Raise DatasetError that names the video when the file does not exist.
    """
    raise NotImplementedError


def probe_video(path: Path) -> tuple[float, int]:
    """Read fps and frame count with OpenCV. Raise DatasetError when the file cannot be opened."""
    raise NotImplementedError


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
    raise NotImplementedError


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
        raise NotImplementedError
