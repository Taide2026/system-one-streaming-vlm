"""Dataset "xd_violence": the XD-Violence test set, 800 videos (500 violent, 300 non-violent).

Asset layout (prepare xd_violence): videos/<name>.mp4 and annotations/annotations.txt. A video's labels are the part
of its name after "_label_": A (non-violent) or up to three of B1 Fighting, B2 Shooting, B4 Riot, B5 Abuse,
B6 Car accident, G Explosion, "0" for none (e.g. "..._label_G-B2-B6.mp4"). annotations.txt has one line per violent
video: its name (most lines without ".mp4", 14 with it), then the first and last frame of each violent event
("<name> s1 e1 [s2 e2 ...]").
"""
from __future__ import annotations

import random
from pathlib import Path

from s1stream.datasets.ucf_crime import NORMAL_CATEGORY, DatasetError, Probe, probe_video
from s1stream.registry import DATASETS
from s1stream.types import VideoItem

ANNOTATION_FILE = Path("annotations") / "annotations.txt"
LABELS = {"B1": "Fighting", "B2": "Shooting", "B4": "Riot", "B5": "Abuse", "B6": "Car accident", "G": "Explosion"}


def classes_of(name: str) -> tuple[str, ...]:
    """The violent classes in a video name, in its order; () for a non-violent (A) video.
    Raise DatasetError for a name without "_label_" or with an unknown code."""
    if "_label_" not in name:
        raise DatasetError(f"No label in video name: {name}")
    codes = [code for code in name.rsplit("_label_", 1)[1].removesuffix(".mp4").split("-") if code != "0"]
    if codes == ["A"]:
        return ()
    unknown = [code for code in codes if code not in LABELS]
    if unknown or not codes:
        raise DatasetError(f"Unknown label {unknown or codes} in video name: {name}")
    return tuple(LABELS[code] for code in codes)


def parse_annotations(text: str) -> dict[str, tuple[tuple[int, int], ...]]:
    """annotations.txt -> {video name with ".mp4": ((first, last), ...)}; a name may come with or without ".mp4".
    Raise DatasetError for a line whose frame numbers do not come in pairs."""
    events = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        frames = [int(value) for value in fields[1:]]
        if not frames or len(frames) % 2:
            raise DatasetError(f"Frames do not come in pairs: {line!r}")
        events[fields[0].removesuffix(".mp4") + ".mp4"] = tuple(zip(frames[0::2], frames[1::2]))
    return events


def select(root: Path, probe: Probe, n_violent: int, n_normal: int, max_duration_s: float, seed: int,
           min_duration_s: float = 0.0) -> list[VideoItem]:
    """Pick n_violent violent and n_normal non-violent test videos with min_duration_s < duration <= max_duration_s,
    as ucf_crime.select does: per group, sort by name, shuffle with random.Random(seed), probe in that order and stop
    at n kept; raise DatasetError naming the group when fewer fit, or when a violent video has no annotation.
    A violent video's category is its classes joined by "+" (e.g. "Explosion+Shooting"), a non-violent one's Normal.
    Return the violent items then the non-violent ones, each group sorted by name."""
    events = parse_annotations((root / ANNOTATION_FILE).read_text())
    names = sorted(path.name for path in (root / "videos").glob("*.mp4"))
    selected: list[VideoItem] = []
    for group, needed in (("violent", n_violent), ("non-violent", n_normal)):
        rows = [name for name in names if bool(classes_of(name)) == (group == "violent")]
        random.Random(seed).shuffle(rows)
        kept: list[VideoItem] = []
        for name in rows:
            if len(kept) == needed:
                break
            path = root / "videos" / name
            fps, n_frames = probe(path)
            if fps <= 0 or not min_duration_s < n_frames / fps <= max_duration_s:
                continue
            classes = classes_of(name)
            if classes and name not in events:
                raise DatasetError(f"No annotation for violent video: {name}")
            kept.append(VideoItem(name, path, "+".join(classes) or NORMAL_CATEGORY, fps, n_frames,
                                  events.get(name, ())))
        if len(kept) < needed:
            raise DatasetError(f"Too few {group} videos fit the duration limits")
        selected.extend(sorted(kept, key=lambda item: item.name))
    return selected


@DATASETS.register("xd_violence")
class XdViolence:
    """Config: {name: xd_violence, asset: xd-violence, n_violent, n_normal, max_duration_s, seed, min_duration_s
    (optional, default 0)}. load_config adds path."""

    def __init__(self, path: str | Path, n_violent: int, n_normal: int, max_duration_s: float, seed: int,
                 asset: str | None = None, probe: Probe = probe_video, min_duration_s: float = 0.0) -> None:
        self.root = Path(path)
        self.n_violent, self.n_normal = n_violent, n_normal
        self.min_duration_s, self.max_duration_s, self.seed = min_duration_s, max_duration_s, seed
        self.probe = probe

    def items(self) -> list[VideoItem]:
        return select(self.root, self.probe, self.n_violent, self.n_normal, self.max_duration_s, self.seed,
                      self.min_duration_s)
