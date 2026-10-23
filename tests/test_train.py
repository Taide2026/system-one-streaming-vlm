"""Tests of the fine-tuning data and loss (s1stream.traindata, s1stream.train), CPU only."""
import json
from pathlib import Path

import pytest
import torch

from s1stream.traindata import (Record, TrainDataError, coverage, draw, read_hivau, read_records, sample_records,
                                split_videos, video_folder, windows_of, write_records)
from s1stream.train import fit_temperature, lr_at, soft_cross_entropy, targets
from s1stream.types import VideoItem, Window


def test_video_folder_from_name():
    assert video_folder("RoadAccidents021_x264") == "RoadAccidents"
    assert video_folder("Abuse001_x264") == "Abuse"
    assert video_folder("Normal_Videos308_x264") == "Training_Normal_Videos_Anomaly"


def test_read_hivau_rounds_events_to_frames_and_checks_files(tmp_path: Path):
    root = tmp_path / "ucf"
    for folder, name in (("Abuse", "Abuse001_x264"), ("Training_Normal_Videos_Anomaly", "Normal_Videos001_x264")):
        (root / "videos" / folder).mkdir(parents=True)
        (root / "videos" / folder / f"{name}.mp4").write_bytes(b"")
    data = {"Abuse001_x264": {"n_frames": 300, "fps": 30.0, "label": ["Abuse"], "events": [[1.0, 2.5]]},
            "Normal_Videos001_x264": {"n_frames": 120, "fps": 30.0, "label": ["Normal"], "events": [[0.5, 3.0]]}}
    json_path = tmp_path / "ucf_database_train.json"
    json_path.write_text(json.dumps(data))
    items = read_hivau(json_path, root)
    assert [item.name for item in items] == ["Abuse001_x264", "Normal_Videos001_x264"]
    assert items[0].events == ((30, 74),)
    assert items[0].category == "Abuse" and items[1].category == "Normal"
    assert items[1].events == ()  # HIVAU's normal segments are not anomaly events
    data["Missing001_x264"] = {"n_frames": 10, "fps": 30.0, "label": ["Missing"], "events": []}
    json_path.write_text(json.dumps(data))
    with pytest.raises(TrainDataError):
        read_hivau(json_path, root)


def _item(events=((30, 74),), n_frames=300):
    return VideoItem("v", Path("v.mp4"), "Abuse", 30.0, n_frames, events)


def test_coverage_is_the_share_of_the_window_inside_events():
    item = _item()  # event 1.0 s - 2.5 s
    assert coverage(Window("v", 0, 0.0, 2.0, (), 1), item) == pytest.approx(0.5)
    assert coverage(Window("v", 1, 2.0, 4.0, (), 1), item) == pytest.approx(0.25)
    assert coverage(Window("v", 2, 4.0, 6.0, (), 0), item) == 0.0
    assert coverage(Window("v", 0, 1.0, 3.0, (), 1), VideoItem("v", Path("v"), "A", 30.0, 300, ((0, 299),))) == 1.0


def test_windows_of_carry_coverage_and_inference_frames():
    records = windows_of(_item(), 2.0, 2.0, 8)
    assert len(records) == 5
    assert records[0].y == pytest.approx(0.5) and records[2].y == 0.0
    assert records[0].frame_indices == tuple(round((0 + (k + 0.5) * 0.25) * 30) for k in range(8))


def test_split_and_draw_are_seeded_and_balanced():
    items = [VideoItem(f"a{i}", Path(f"a{i}"), "Abuse", 30.0, 600, ((0, 299),)) for i in range(10)]
    items += [VideoItem(f"n{i}", Path(f"n{i}"), "Normal", 30.0, 600, ()) for i in range(10)]
    train, held = split_videos(items, 0.2, seed=3)
    assert len(held) == 4 and len(train) == 16 and split_videos(items, 0.2, seed=3) == (train, held)
    assert sum(item.category == "Normal" for item in held) == 2
    train_records, val_records = sample_records(items, 2.0, 2.0, 8, 20, 20, 10, 0.2, seed=3)
    assert len(train_records) == 40 and sum(r.y > 0 for r in train_records) == 20
    assert len(val_records) == 10 and sum(r.y > 0 for r in val_records) == 5
    train_names = {r.video for r in train_records}
    assert train_names.isdisjoint({r.video for r in val_records})
    with pytest.raises(TrainDataError):
        draw(train_records, 1000, 0, 0)


def test_records_round_trip(tmp_path: Path):
    records = [Record("v", "v.mp4", "Abuse", 0, 0.0, 2.0, (1, 2, 3), 0.5)]
    write_records(records, tmp_path / "r.jsonl")
    assert read_records(tmp_path / "r.jsonl") == records


def test_soft_cross_entropy_and_targets():
    records = [Record("v", "v.mp4", "A", 0, 0.0, 2.0, (0,), 0.75), Record("v", "v.mp4", "A", 1, 2.0, 4.0, (0,), 0.0)]
    target = targets(records, ("No", "Yes"), "Yes")  # the release sorts options by id
    assert target.tolist() == [[0.25, 0.75], [1.0, 0.0]]
    logits = torch.log(torch.tensor([[0.25, 0.75], [1.0, 1e-9]]))
    assert soft_cross_entropy(logits, target).item() == pytest.approx(
        (-(0.25 * torch.log(torch.tensor(0.25)) + 0.75 * torch.log(torch.tensor(0.75))).item() + 0.0) / 2, abs=1e-5)


def test_fit_temperature_recovers_a_known_scale():
    torch.manual_seed(0)
    true_logits = torch.randn(400, 2) * 2
    target = true_logits.softmax(dim=-1)
    assert fit_temperature(true_logits * 2.0, target) == pytest.approx(2.0, rel=0.05)
    assert fit_temperature(true_logits / 2.0, target) == pytest.approx(0.5, rel=0.05)


def test_lr_schedule_warms_up_then_decays():
    assert lr_at(0, 100, 10) == pytest.approx(0.1)
    assert lr_at(9, 100, 10) == pytest.approx(1.0)
    assert lr_at(100, 100, 10) == pytest.approx(0.1)
    assert lr_at(55, 100, 10) > lr_at(90, 100, 10)
