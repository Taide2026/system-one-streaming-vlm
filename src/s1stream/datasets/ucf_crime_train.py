"""Dataset "ucf_crime_train": UCF-Crime train videos with the Holmes-VAU events (finetune-layer-probing decision 4).

Asset layout: the ucf-crime asset (see ucf_crime.py) gives <root>/splits/Anomaly_Train.txt (lines such as
"Abuse/Abuse001_x264.mp4" or "Training_Normal_Videos_Anomaly/Normal_Videos001_x264.mp4") and <root>/videos/<line>.
The holmes-vau-ucf-train asset gives ucf_database_train.json: {"Abuse001_x264": {"events": [[3.933, 15.867], ...],
"fps": 30.0, "n_frames": 2729, ...}, ...}, events in seconds. For normal videos Holmes lists caption segments as
"events"; they are not anomalies.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from s1stream.datasets.ucf_crime import DatasetError, Probe, probe_video  # noqa: F401
from s1stream.registry import DATASETS
from s1stream.types import VideoItem

SPLIT_FILE = "splits/Anomaly_Train.txt"
HOLMES_FILE = "ucf_database_train.json"
NORMAL_TRAIN_FOLDER = "Training_Normal_Videos_Anomaly"
NORMAL_CATEGORY = "Normal"


def holmes_frames(events_s: list[list[float]], fps: float) -> tuple[tuple[int, int], ...]:
    """Holmes events in seconds -> inclusive frame ranges (round(start * fps), round(end * fps) - 1), in order."""
    raise NotImplementedError


@DATASETS.register("ucf_crime_train")
class UcfCrimeTrain:
    """Config: {name: ucf_crime_train, asset: ucf-crime, holmes_asset: holmes-vau-ucf-train, max_duration_s,
    per_class, n_normal, seed}. load_config adds path (the ucf-crime folder); holmes_asset is resolved here with
    resolver (default s1stream.assets.resolve)."""

    def __init__(self, path: str | Path, holmes_asset: str, max_duration_s: float, per_class: int, n_normal: int,
                 seed: int, asset: str | None = None, probe: Probe = probe_video,
                 resolver: Callable[[str], Path] | None = None) -> None:
        raise NotImplementedError

    def items(self) -> list[VideoItem]:
        """The selected train videos.

        1. Read SPLIT_FILE. Keep a line when the Holmes JSON has its stem (file name without ".mp4").
        2. Probe each kept video (root / "videos" / line). Drop it when fps <= 0 or n_frames / fps > max_duration_s.
        3. Category: NORMAL_CATEGORY for lines in NORMAL_TRAIN_FOLDER, else the folder name.
        4. Group by category; sort each group by name; pick random.Random(seed).sample(group, n) with
           n = n_normal for NORMAL_CATEGORY and per_class for every other category (one new Random per group).
           Raise DatasetError that names the category when the group has fewer than n videos.
        5. VideoItem(name=<file name with .mp4>, path, category, fps, n_frames, events): events =
           holmes_frames(holmes[stem]["events"], fps) for anomaly videos and () for normal videos.
        Return the anomaly categories in sorted order, then the normal videos; each group sorted by name.
        """
        raise NotImplementedError
