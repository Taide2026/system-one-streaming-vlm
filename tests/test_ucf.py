"""Unit ucf: ucf_crime dataset, windows, frames input form."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from s1stream.datasets.ucf_crime import (AnnotationRow, DatasetError, UcfCrime, parse_annotation_line, select,
                                         video_path)
from s1stream.input_forms.frames import FrameReadError, Frames
from s1stream.types import Question, VideoItem
from s1stream.windows import plan_windows

QUESTION = Question(state="s", text="q", options=("Yes", "No"), positive="Yes")


def make_tree(root: Path, rows: list[AnnotationRow]) -> None:
    for row in rows:
        folder = "Testing_Normal_Videos_Anomaly" if row.category == "Normal" else row.category
        (root / "videos" / folder).mkdir(parents=True, exist_ok=True)
        (root / "videos" / folder / row.name).touch()


def forty_rows() -> list[AnnotationRow]:
    rows = [AnnotationRow(f"Abuse{i:03d}_x264.mp4", "Abuse", ((10, 20),)) for i in range(20)]
    rows += [AnnotationRow(f"Normal_Videos_{i:03d}_x264.mp4", "Normal", ()) for i in range(20)]
    return rows


def probe_by_index(path: Path) -> tuple[float, int]:
    # Even-numbered videos are 30 s, odd-numbered videos are 90 s.
    index = int(path.name.split("_")[-2][-3:]) if "Normal" in path.name else int(path.name[5:8])
    seconds = 30 if index % 2 == 0 else 90
    return 30.0, seconds * 30


def item(seconds: float, events=()) -> VideoItem:
    return VideoItem(name="v", path=Path("v.mp4"), category="X", fps=30.0, n_frames=int(seconds * 30),
                     events=tuple(events))


def test_parse_one_event_row():
    row = parse_annotation_line("Abuse028_x264.mp4  Abuse  165  240  -1  -1  ")
    assert (row.name, row.category, row.events) == ("Abuse028_x264.mp4", "Abuse", ((165, 240),))


def test_parse_two_event_row():
    row = parse_annotation_line("Arson011_x264.mp4  Arson  150  420  680  1267")
    assert row.events == ((150, 420), (680, 1267))


def test_parse_normal_row():
    row = parse_annotation_line("Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1")
    assert row.category == "Normal"
    assert row.events == ()


def test_select_ten_and_ten_within_length_limit(tmp_path):
    rows = forty_rows()
    make_tree(tmp_path, rows)
    items = select(rows, tmp_path, probe_by_index, 10, 10, 60, seed=20261009)
    assert [i.category == "Normal" for i in items] == [False] * 10 + [True] * 10
    assert all(i.duration_s <= 60 for i in items)
    assert items[0].events == ((10, 20),)


def test_select_same_seed_same_list_other_seed_other_list(tmp_path):
    rows = forty_rows()
    make_tree(tmp_path, rows)
    first = [i.name for i in select(rows, tmp_path, probe_by_index, 5, 5, 60, seed=1)]
    again = [i.name for i in select(rows, tmp_path, probe_by_index, 5, 5, 60, seed=1)]
    other = [i.name for i in select(rows, tmp_path, probe_by_index, 5, 5, 60, seed=2)]
    assert first == again
    assert first != other


def test_select_too_few_short_videos_raises(tmp_path):
    rows = forty_rows()
    make_tree(tmp_path, rows)
    with pytest.raises(DatasetError):
        select(rows, tmp_path, probe_by_index, 11, 10, 60, seed=1)


def test_video_path_class_folder_normal_folder_and_missing(tmp_path):
    rows = [AnnotationRow("Abuse028_x264.mp4", "Abuse", ((1, 2),)),
            AnnotationRow("Normal_Videos_003_x264.mp4", "Normal", ())]
    make_tree(tmp_path, rows)
    assert video_path(tmp_path, rows[0]) == tmp_path / "videos" / "Abuse" / "Abuse028_x264.mp4"
    assert video_path(tmp_path, rows[1]) == tmp_path / "videos" / "Testing_Normal_Videos_Anomaly" / rows[1].name
    with pytest.raises(DatasetError) as err:
        video_path(tmp_path, AnnotationRow("Arson001_x264.mp4", "Arson", ()))
    assert "Arson001_x264.mp4" in str(err.value)


def test_plan_windows_ten_seconds():
    wins = plan_windows(item(10.0), 2.0, 2.0, 8)
    assert [(w.start_s, w.end_s) for w in wins] == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    assert [w.index for w in wins] == [0, 1, 2, 3, 4]


def test_plan_windows_drops_partial_last_window():
    assert len(plan_windows(item(11.0), 2.0, 2.0, 8)) == 5


def test_frame_indices_even_inside_increasing():
    win = plan_windows(item(10.0), 2.0, 2.0, 8)[1]
    idx = win.frame_indices
    assert len(idx) == 8
    assert list(idx) == sorted(set(idx))
    assert 60 <= idx[0] and idx[-1] <= 119
    gaps = np.diff(idx)
    assert gaps.max() - gaps.min() <= 1


def test_label_overlap_one_frame_and_one_frame_before():
    # Window 0-2 s covers frames 0-59.
    assert plan_windows(item(4.0, [(59, 80)]), 2.0, 2.0, 8)[0].label == 1
    assert plan_windows(item(4.0, [(60, 80)]), 2.0, 2.0, 8)[0].label == 0


def test_labels_zero_without_events():
    assert {w.label for w in plan_windows(item(10.0), 2.0, 2.0, 8)} == {0}


def synthetic_video(path: Path, n_frames: int = 60) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for i in range(n_frames):
        frame = np.full((240, 320, 3), i * 4 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_frames_input_form_returns_eight_rgb_images(tmp_path):
    path = tmp_path / "v.mp4"
    synthetic_video(path)
    video = VideoItem(name="v.mp4", path=path, category="X", fps=30.0, n_frames=60)
    win = plan_windows(video, 2.0, 2.0, 8)[0]
    inp = Frames().build(video, win, QUESTION)
    assert len(inp.images) == 8
    assert all(img.mode == "RGB" and img.size == (320, 240) for img in inp.images)
    assert inp.text is None


def test_frames_input_form_raises_past_end(tmp_path):
    path = tmp_path / "v.mp4"
    synthetic_video(path)
    video = VideoItem(name="v.mp4", path=path, category="X", fps=30.0, n_frames=60)
    win = plan_windows(VideoItem("v.mp4", path, "X", 30.0, 300), 2.0, 2.0, 8)[4]  # frames 240-299 do not exist
    with pytest.raises(FrameReadError):
        Frames().build(video, win, QUESTION)


def test_ucf_crime_items_reads_annotation_file_and_selects(tmp_path):
    rows = forty_rows()
    make_tree(tmp_path, rows)
    (tmp_path / "annotations").mkdir()
    lines = [f"{r.name}  {r.category}  {r.events[0][0] if r.events else -1}  {r.events[0][1] if r.events else -1}  -1  -1"
             for r in rows]
    (tmp_path / "annotations" / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").write_text("\n".join(lines) + "\n")
    dataset = UcfCrime(path=tmp_path, n_anomaly=3, n_normal=2, max_duration_s=60, seed=7, probe=probe_by_index)
    expected = select(rows, tmp_path, probe_by_index, 3, 2, 60, seed=7)
    assert dataset.items() == expected
