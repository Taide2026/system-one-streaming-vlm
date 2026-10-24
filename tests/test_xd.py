"""Unit xd: the xd_violence dataset and its asset prepare step, on a fake tree (no videos are decoded)."""
import io
from pathlib import Path

import pytest

from s1stream import assets
from s1stream.datasets.ucf_crime import DatasetError
from s1stream.datasets.xd_violence import XdViolence, classes_of, parse_annotations, select

VIOLENT = ["Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6.mp4", "v=abc__#1_label_B1-0-0.mp4",
           "v=def__#1_label_B4-0-0.mp4"]
NORMAL = ["A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A.mp4", "Your.Name.2016__#00-05-52_00-08-22_label_A.mp4"]
ANNOTATIONS = ("Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6 10 20\n"
               "v=abc__#1_label_B1-0-0 523 610 900 1700\n"
               "v=def__#1_label_B4-0-0 0 99\n")


def make_tree(root: Path, annotations: str = ANNOTATIONS) -> None:
    (root / "videos").mkdir(parents=True)
    for name in VIOLENT + NORMAL:
        (root / "videos" / name).touch()
    (root / "annotations").mkdir()
    (root / "annotations" / "annotations.txt").write_text(annotations)


def probe_60s(path: Path) -> tuple[float, int]:
    return 24.0, 24 * 60


def test_classes_of_reads_every_code_and_a_as_none():
    assert classes_of(VIOLENT[0]) == ("Explosion", "Shooting", "Car accident")
    assert classes_of(VIOLENT[1]) == ("Fighting",)
    assert classes_of(NORMAL[0]) == ()
    with pytest.raises(DatasetError):
        classes_of("v=x__#1_label_B9-0-0.mp4")


def test_parse_annotations_pairs_frames_per_video():
    events = parse_annotations(ANNOTATIONS)
    assert events["v=abc__#1_label_B1-0-0.mp4"] == ((523, 610), (900, 1700))
    with pytest.raises(DatasetError):
        parse_annotations("v=abc__#1_label_B1-0-0 523 610 900\n")
    # Some lines of the authors' file name the video with ".mp4".
    assert parse_annotations("v=abc__#1_label_B1-0-0.mp4 3 9\n") == {"v=abc__#1_label_B1-0-0.mp4": ((3, 9),)}


def test_select_splits_violent_and_normal_with_events(tmp_path):
    make_tree(tmp_path)
    items = select(tmp_path, probe_60s, 3, 2, 4000, seed=1)
    assert [item.category for item in items] == ["Explosion+Shooting+Car accident", "Fighting", "Riot",
                                                 "Normal", "Normal"]
    assert items[1].events == ((523, 610), (900, 1700))
    assert items[3].events == ()


def test_select_same_seed_same_list_and_duration_limits(tmp_path):
    make_tree(tmp_path)
    first = [item.name for item in select(tmp_path, probe_60s, 2, 1, 4000, seed=3)]
    assert first == [item.name for item in select(tmp_path, probe_60s, 2, 1, 4000, seed=3)]
    with pytest.raises(DatasetError, match="violent"):
        select(tmp_path, probe_60s, 1, 1, 30, seed=1)


def test_select_raises_for_a_violent_video_without_annotation(tmp_path):
    make_tree(tmp_path, annotations=ANNOTATIONS.splitlines(keepends=True)[0])
    with pytest.raises(DatasetError, match="No annotation"):
        select(tmp_path, probe_60s, 3, 0, 4000, seed=1)


def test_xd_violence_items_reads_the_asset_folder(tmp_path):
    make_tree(tmp_path)
    dataset = XdViolence(path=tmp_path, n_violent=3, n_normal=2, max_duration_s=4000, seed=7, probe=probe_60s)
    assert [item.name for item in dataset.items()] == [item.name for item in select(tmp_path, probe_60s, 3, 2, 4000, 7)]


def test_prepare_xd_violence_adds_the_annotations(tmp_path, monkeypatch):
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda url: io.BytesIO(ANNOTATIONS.encode()))
    assets.prepare_xd_violence(tmp_path)
    assert (tmp_path / "annotations" / "annotations.txt").read_text() == ANNOTATIONS


def test_xd_names_selects_in_order(tmp_path):
    from s1stream.datasets.xd_violence import XdViolence
    root = tmp_path
    (root / "videos").mkdir(); (root / "annotations").mkdir()
    for name in ("a_label_A.mp4", "b_label_B1.mp4"):
        (root / "videos" / name).write_bytes(b"")
    (root / "annotations" / "annotations.txt").write_text("b_label_B1.mp4 10 20\n")
    items = XdViolence(root, names=["b_label_B1.mp4", "a_label_A.mp4"], probe=lambda p: (24.0, 240)).items()
    assert [i.name for i in items] == ["b_label_B1.mp4", "a_label_A.mp4"]
    assert items[0].events == ((10, 20),) and items[0].category == "Fighting" and items[1].category == "Normal"
