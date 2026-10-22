"""Unit train-dataset: the ucf_crime_train dataset (finetune-layer-probing decision 4)."""
import json
from pathlib import Path

import pytest

from s1stream.datasets.ucf_crime import DatasetError
from s1stream.datasets.ucf_crime_train import UcfCrimeTrain

LONG = {"Abuse003_x264", "Normal_Videos003_x264"}  # these two are 200 s long; every other video is 60 s


def make_assets(tmp_path: Path, n_abuse: int = 6, n_burglary: int = 6, n_normal: int = 6,
                missing: tuple[str, ...] = ()) -> tuple[Path, Path]:
    """A ucf-crime folder (split file + empty video files) and a Holmes folder; returns (ucf root, holmes folder)."""
    root, holmes_dir = tmp_path / "ucf", tmp_path / "holmes"
    lines = [f"Abuse/Abuse{i:03d}_x264.mp4" for i in range(n_abuse)]
    lines += [f"Burglary/Burglary{i:03d}_x264.mp4" for i in range(n_burglary)]
    lines += [f"Training_Normal_Videos_Anomaly/Normal_Videos{i:03d}_x264.mp4" for i in range(n_normal)]
    holmes = {}
    for line in lines:
        (root / "videos" / line).parent.mkdir(parents=True, exist_ok=True)
        (root / "videos" / line).touch()
        stem = Path(line).stem
        if stem not in missing:
            holmes[stem] = {"events": [[1.0, 2.0], [10.5, 12.25]], "fps": 30.0, "n_frames": 1800}
    (root / "splits").mkdir(parents=True)
    (root / "splits" / "Anomaly_Train.txt").write_text("\n".join(lines) + "\n")
    holmes_dir.mkdir()
    (holmes_dir / "ucf_database_train.json").write_text(json.dumps(holmes))
    return root, holmes_dir


def probe(path: Path) -> tuple[float, int]:
    return 30.0, (200 if path.stem in LONG else 60) * 30


def dataset(root: Path, holmes_dir: Path, per_class: int = 2, n_normal: int = 2, seed: int = 7,
            max_duration_s: float = 128.0) -> UcfCrimeTrain:
    return UcfCrimeTrain(root, holmes_asset="holmes-vau-ucf-train", max_duration_s=max_duration_s,
                         per_class=per_class, n_normal=n_normal, seed=seed, probe=probe,
                         resolver=lambda name: holmes_dir)


def test_keeps_only_videos_with_holmes_entry(tmp_path):
    missing = tuple(f"Abuse{i:03d}_x264" for i in range(4)) + ("Burglary000_x264",)
    root, holmes_dir = make_assets(tmp_path, missing=missing)
    names = {item.name for item in dataset(root, holmes_dir).items()}
    assert {"Abuse004_x264.mp4", "Abuse005_x264.mp4"} <= names
    assert not any(Path(name).stem in missing for name in names)


def test_drops_videos_longer_than_limit(tmp_path):
    root, holmes_dir = make_assets(tmp_path, n_abuse=4, n_normal=4)
    names = {item.name for item in dataset(root, holmes_dir, per_class=3, n_normal=3).items()}
    assert "Abuse003_x264.mp4" not in names and "Normal_Videos003_x264.mp4" not in names
    assert {"Abuse000_x264.mp4", "Abuse001_x264.mp4", "Abuse002_x264.mp4"} <= names


def test_picks_counts_with_seed_sorted(tmp_path):
    root, holmes_dir = make_assets(tmp_path)
    first = dataset(root, holmes_dir, per_class=2, n_normal=3, seed=11).items()
    again = dataset(root, holmes_dir, per_class=2, n_normal=3, seed=11).items()
    assert first == again
    counts = {}
    for item in first:
        counts[item.category] = counts.get(item.category, 0) + 1
    assert counts == {"Abuse": 2, "Burglary": 2, "Normal": 3}
    assert [item.category for item in first] == ["Abuse", "Abuse", "Burglary", "Burglary", "Normal", "Normal", "Normal"]
    for category in counts:
        names = [item.name for item in first if item.category == category]
        assert names == sorted(names)


def test_converts_event_seconds_to_frames(tmp_path):
    root, holmes_dir = make_assets(tmp_path)
    abuse = next(item for item in dataset(root, holmes_dir).items() if item.category == "Abuse")
    assert (abuse.fps, abuse.n_frames) == (30.0, 1800)
    assert abuse.events == ((30, 59), (315, 367))  # round(1.0*30), round(2.0*30)-1; round(10.5*30), round(12.25*30)-1


def test_normal_videos_get_no_events(tmp_path):
    root, holmes_dir = make_assets(tmp_path)
    normal = [item for item in dataset(root, holmes_dir).items() if item.category == "Normal"]
    assert normal and all(item.events == () for item in normal)


def test_too_few_videos_names_class(tmp_path):
    root, holmes_dir = make_assets(tmp_path, n_burglary=1)
    with pytest.raises(DatasetError) as err:
        dataset(root, holmes_dir, per_class=2).items()
    assert "Burglary" in str(err.value)
