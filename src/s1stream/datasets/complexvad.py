"""Dataset "complexvad": ComplexVAD test videos (Zenodo 15707073, CC-BY-SA-4.0), a high-resolution fixed-camera test (2026-11-13).

The native videos are 1920x1080. scripts/cvad-prep.py writes, per resolution, <root>/<variant>/<id>.avi holding only the
frames our windows sample (8 fps, each sampled frame twice; stored frame j is native time j / 8 s) and
<root>/meta/<id>.json with the native fps. Annotations: <root>/annotations/<id>_annotation.json,
{total_frames, annotations: [{track_id, frame_id, bbox [x, y, w, h], object_type}]}; a native frame with any box is
anomalous, and runs of such frames are the events (converted to stored-frame units here).
Every test video has an anomaly; there are no normal videos. Config: {name: complexvad, path: <root>, variant: v180|v360|v720,
n_videos (optional, first n by name), videos (optional: a file under <root> listing the ids to use)}.
"""
from __future__ import annotations

import json
from pathlib import Path

from s1stream.datasets.ucf_crime import DatasetError
from s1stream.registry import DATASETS
from s1stream.types import VideoItem


def native_events(path: Path, gap: int = 1) -> tuple[tuple[int, int], ...]:
    """Inclusive runs of native frames with at least one box; runs closer than `gap` frames are one event."""
    frames = sorted({int(a["frame_id"]) for a in json.loads(path.read_text())["annotations"]})
    runs: list[list[int]] = []
    for f in frames:
        if runs and f <= runs[-1][1] + gap:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    return tuple((a, b) for a, b in runs)


@DATASETS.register("complexvad")
class ComplexVad:
    def __init__(self, path: str | Path, variant: str = "v180", n_videos: int | None = None, asset: str | None = None,
                 videos: str | None = None) -> None:
        self.root, self.variant, self.n_videos = Path(path), variant, n_videos
        self.videos = set((self.root / videos).read_text().split()) if videos else None      # a fixed list of ids, one per line

    def items(self) -> list[VideoItem]:
        paths = sorted((self.root / self.variant).glob("[0-9]*.avi"))
        paths = [p for p in paths if not p.name.endswith(".tmp.avi") and (self.videos is None or p.stem in self.videos)]
        if self.n_videos is not None:
            paths = paths[:self.n_videos]
        if not paths:
            raise DatasetError(f"No prepared videos in {self.root / self.variant}")
        items = []
        for p in paths:
            meta = json.loads((self.root / "meta" / f"{p.stem}.json").read_text())
            fps = meta["fps"]
            events = tuple((int(a / fps * 8), int(b / fps * 8)) for a, b in native_events(self.root / "annotations" / f"{p.stem}_annotation.json"))
            items.append(VideoItem(p.stem, p, "abnormal", 8.0, int(meta["stored_frames"]), events))
        return items
