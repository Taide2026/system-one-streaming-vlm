"""Dataset "ucf101": UCF101 action clips (Soomro et al., 2012), the generalization test of the fine-tunes (2026-10-28).

A clip has no events: every window is label 0 and the manifest's category is the action class; accuracy comes from
the option probabilities offline (scripts/ucf101-accuracy.py), the runner's AUC / AP are None. Layout after unpacking
the official rar and split zip: <root>/UCF-101/<Class>/v_<Class>_gXX_cYY.avi, <root>/ucfTrainTestlist/testlist01.txt
(lines "<Class>/v_..avi"). Config: {name: ucf101, path: <root>, split: testlist01.txt, n_videos (optional: a seeded
sample of the split), seed}.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from s1stream.datasets.ucf_crime import DatasetError, Probe, probe_video
from s1stream.registry import DATASETS
from s1stream.types import VideoItem

VIDEOS_DIR = "UCF-101"
SPLITS_DIR = "ucfTrainTestlist"


def read_split(path: Path) -> list[tuple[str, str]]:
    """(class, relative path) per non-empty line of a split list; a train list's trailing class index is dropped."""
    rows: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rel = line.split()[0]
        if "/" not in rel:
            raise DatasetError(f"Malformed split line: {line!r}")
        rows.append((rel.split("/")[0], rel))
    return rows


def select(rows: Sequence[tuple[str, str]], root: Path, probe: Probe, n_videos: int | None, seed: int) -> list[VideoItem]:
    """All rows (sorted by path), or a seeded sample of n_videos of them; each probed for fps and frame count."""
    rows = sorted(rows, key=lambda r: r[1])
    if n_videos is not None:
        rows = sorted(random.Random(seed).sample(list(rows), n_videos), key=lambda r: r[1])
    items: list[VideoItem] = []
    for category, rel in rows:
        path = root / VIDEOS_DIR / rel
        if not path.is_file():
            raise DatasetError(f"Video not found: {rel} ({path})")
        fps, n_frames = probe(path)
        if fps <= 0:
            raise DatasetError(f"Cannot read fps: {path}")
        items.append(VideoItem(Path(rel).name, path, category, fps, n_frames, ()))
    return items


@DATASETS.register("ucf101")
class Ucf101:
    """Config: {name: ucf101, path: <root>, split: testlist01.txt, n_videos (optional), seed (default 0)}."""

    def __init__(self, path: str | Path, split: str = "testlist01.txt", n_videos: int | None = None, seed: int = 0,
                 asset: str | None = None, probe: Probe = probe_video) -> None:
        self.root = Path(path)
        self.split, self.n_videos, self.seed, self.probe = split, n_videos, int(seed), probe

    def items(self) -> list[VideoItem]:
        return select(read_split(self.root / SPLITS_DIR / self.split), self.root, self.probe, self.n_videos, self.seed)
