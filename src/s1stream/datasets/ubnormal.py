"""Dataset "ubnormal": UBnormal (Acsintoae et al., CVPR 2022), the transfer test of the fine-tunes (2026-10-29).

Layout of the hub zip (Dxuan77/UBnormal): <root>/Scene<n>/<name>.mp4 with <name> = (normal|abnormal)_scene_<n>_scenario_<k>,
and for abnormal videos <root>/Scene<n>/<name>_annotations/<name>_tracks.txt (lines "track id, first frame, last frame",
inclusive), the frame-level ground truth of the authors' write_frame_level_ground_truth.py. Split lists from the
authors' repo (scripts/{abnormal,normal}_<split>_video_names.txt) at <root>/splits/. Config: {name: ubnormal, path: <root>,
split: test, n_videos (optional seeded sample), seed}. Test: 157 abnormal + 52 normal videos. CC BY-NC-ND 4.0.
"""
from __future__ import annotations

import random
from pathlib import Path

from s1stream.datasets.ucf_crime import DatasetError, Probe, probe_video
from s1stream.registry import DATASETS
from s1stream.types import VideoItem


def read_tracks(path: Path) -> tuple[tuple[int, int], ...]:
    """Inclusive (first, last) frame ranges of the abnormal tracks, merged where they overlap."""
    spans = []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 3:
            spans.append((int(float(parts[1])), int(float(parts[2]))))
    spans.sort()
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return tuple((a, b) for a, b in merged)


def video_path(root: Path, name: str) -> Path:
    scene = name.split("_")[2]
    return root / f"Scene{scene}" / f"{name}.mp4"


@DATASETS.register("ubnormal")
class Ubnormal:
    """Config: {name: ubnormal, path: <root>, split: test, n_videos (optional), seed (default 0)}."""

    def __init__(self, path: str | Path, split: str = "test", n_videos: int | None = None, seed: int = 0,
                 asset: str | None = None, probe: Probe = probe_video) -> None:
        self.root, self.split, self.n_videos, self.seed, self.probe = Path(path), split, n_videos, int(seed), probe

    def items(self) -> list[VideoItem]:
        names = []
        for kind in ("abnormal", "normal"):
            lst = self.root / "splits" / f"{kind}_{self.split}_video_names.txt"
            if not lst.is_file():
                raise DatasetError(f"Split list not found: {lst}")
            names += [(kind, n.strip()) for n in lst.read_text().splitlines() if n.strip()]
        names.sort(key=lambda t: t[1])
        if self.n_videos is not None:
            names = sorted(random.Random(self.seed).sample(names, self.n_videos), key=lambda t: t[1])
        items = []
        for kind, name in names:
            path = video_path(self.root, name)
            if not path.is_file():
                raise DatasetError(f"Video not found: {name} ({path})")
            events = read_tracks(path.parent / f"{name}_annotations" / f"{name}_tracks.txt") if kind == "abnormal" else ()
            fps, n_frames = self.probe(path)
            items.append(VideoItem(name, path, kind, fps, n_frames, events))
        return items
