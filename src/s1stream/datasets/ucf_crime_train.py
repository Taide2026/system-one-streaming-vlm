"""Dataset "ucf_crime_train": UCF-Crime train videos with the Holmes-VAU events (finetune-layer-probing decision 4).

Asset layout: the ucf-crime asset (see ucf_crime.py) gives <root>/splits/Anomaly_Train.txt (lines such as
"Abuse/Abuse001_x264.mp4" or "Training_Normal_Videos_Anomaly/Normal_Videos001_x264.mp4") and <root>/videos/<line>.
The holmes-vau-ucf-train asset gives ucf_database_train.json: {"Abuse001_x264": {"events": [[3.933, 15.867], ...],
"fps": 30.0, "n_frames": 2729, ...}, ...}, events in seconds. For normal videos Holmes lists caption segments as
"events"; they are not anomalies.
"""
from __future__ import annotations

import json
import random
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
    return tuple((round(start * fps), round(end * fps) - 1) for start, end in events_s)


@DATASETS.register("ucf_crime_train")
class UcfCrimeTrain:
    """Config: {name: ucf_crime_train, asset: ucf-crime, holmes_asset: holmes-vau-ucf-train, max_duration_s,
    per_class, n_normal, seed}. load_config adds path (the ucf-crime folder); holmes_asset is resolved here with
    resolver (default s1stream.assets.resolve)."""

    def __init__(self, path: str | Path, holmes_asset: str, max_duration_s: float, per_class: int, n_normal: int,
                 seed: int, asset: str | None = None, probe: Probe = probe_video,
                 resolver: Callable[[str], Path] | None = None) -> None:
        self.root = Path(path)
        self.max_duration_s, self.per_class, self.n_normal, self.seed = max_duration_s, per_class, n_normal, seed
        self.probe = probe
        if resolver is None:
            from s1stream.assets import resolve
            resolver = resolve
        self.holmes_path = resolver(holmes_asset) / HOLMES_FILE
        try:
            with self.holmes_path.open() as stream:
                self.holmes = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(f"Cannot read Holmes data: {self.holmes_path}") from exc

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
        groups: dict[str, list[VideoItem]] = {}
        for line in (self.root / SPLIT_FILE).read_text().splitlines():
            if not line.strip():
                continue
            relative = Path(line)
            name, stem = relative.name, relative.stem
            if stem not in self.holmes:
                continue
            path = self.root / "videos" / relative
            fps, n_frames = self.probe(path)
            if fps <= 0 or n_frames / fps > self.max_duration_s:
                continue
            category = NORMAL_CATEGORY if relative.parent.name == NORMAL_TRAIN_FOLDER else relative.parent.name
            events = () if category == NORMAL_CATEGORY else holmes_frames(self.holmes[stem]["events"], fps)
            groups.setdefault(category, []).append(VideoItem(name, path, category, fps, n_frames, events))

        selected: list[VideoItem] = []
        categories = sorted(category for category in groups if category != NORMAL_CATEGORY)
        if NORMAL_CATEGORY in groups:
            categories.append(NORMAL_CATEGORY)
        for category in categories:
            group = sorted(groups[category], key=lambda item: item.name)
            needed = self.n_normal if category == NORMAL_CATEGORY else self.per_class
            if len(group) < needed:
                raise DatasetError(f"Too few {category} videos")
            picked = random.Random(self.seed).sample(group, needed)
            selected.extend(sorted(picked, key=lambda item: item.name))
        return selected
