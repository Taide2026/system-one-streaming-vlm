"""Dataset "msad": MSAD (Zhu et al., NeurIPS 2024; gated, research use only), Protocol-ii test split: 120 abnormal and
120 normal videos from fixed cameras in 14 scenarios (2026-11-13). A third dataset for the resolution study.

scripts/msad-prep.py writes <root>/prep/<variant>/<name>.avi (variant fixes the frame height: v180, v360, v720; aspect
ratio kept) holding only the frames our windows sample (8 fps, each sampled frame twice; stored frame j is native time
j / 8 s), and <root>/prep/meta/<name>.json with the native fps. Annotations: <root>/anomaly_annotation.csv, one row per
abnormal video: name, scenario, total frames, first and last anomalous frame (native, inclusive).
No boxes. A video whose source is lower than the variant's height is absent from that variant; `videos_from` restricts
the items to the videos another variant also has, so that two variants are compared on the same videos.
Config: {name: msad, path: <root>, variant: v180|v360|v720, videos_from: <variant> (optional)}.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from s1stream.datasets.ucf_crime import DatasetError
from s1stream.registry import DATASETS
from s1stream.types import VideoItem


@DATASETS.register("msad")
class Msad:
    def __init__(self, path: str | Path, variant: str = "v180", videos_from: str | None = None, asset: str | None = None) -> None:
        self.root, self.variant, self.videos_from = Path(path), variant, videos_from

    def items(self) -> list[VideoItem]:
        prep = self.root / "prep"
        paths = sorted(p for p in (prep / self.variant).glob("*.avi") if not p.name.endswith(".tmp.avi"))
        if self.videos_from:
            keep = {p.stem for p in (prep / self.videos_from).glob("*.avi")}
            paths = [p for p in paths if p.stem in keep]
        if not paths:
            raise DatasetError(f"No prepared videos in {prep / self.variant}")
        spans = {}
        with (self.root / "anomaly_annotation.csv").open() as f:
            for row in csv.reader(f):
                if row and row[0] != "name" and len(row) >= 5 and row[3].strip() and row[4].strip():
                    spans[row[0].strip()] = (int(float(row[3])), int(float(row[4])), row[1].strip())
        items = []
        for p in paths:
            meta = json.loads((prep / "meta" / f"{p.stem}.json").read_text()); fps = meta["fps"]
            if p.stem in spans:
                a, b, scen = spans[p.stem]
                events = ((int(a / fps * 8), int(b / fps * 8)),); cat = p.stem.rsplit("_", 1)[0]
            else:
                events = (); cat = "Normal"
            items.append(VideoItem(p.stem, p, cat, 8.0, int(meta["stored_frames"]), events))
        return items
