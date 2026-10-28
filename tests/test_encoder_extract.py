"""Unit encoder-extract: encoder config, feature extraction for main and head folders, run.json, CLI."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from s1stream import cli
from s1stream.config import ConfigError, WindowSpec
from s1stream.encoder.adapters import EncoderOutput
from s1stream.encoder.extract import (SETS, EncoderConfig, WindowRow, load_encoder_config, plan_sets,
                                      run_encoder_extract, window_meta)
from s1stream.types import VideoItem

WINDOW = WindowSpec(length_s=2.0, stride_s=2.0, frames=8)
STATES, DIMS, HEAD_DIMS = 3, 4, 5


def resolver(name: str) -> Path:
    return Path("/assets") / name


def video(name: str, category: str, seconds: float, events=()) -> VideoItem:
    return VideoItem(name=name, path=Path(name), category=category, fps=10.0, n_frames=int(seconds * 10),
                     events=tuple(events))


def config_data() -> dict:
    return {
        "name": "encoder_probe_ucf_short_v1_f8",
        "train": {
            "name": "ucf_crime_train",
            "asset": "ucf-crime",
            "holmes_asset": "holmes-vau-ucf-train",
            "max_duration_s": 128.0,
            "per_class": 15,
            "n_normal": 195,
            "seed": 20261022,
        },
        "test": {
            "name": "ucf_crime",
            "asset": "ucf-crime",
            "n_anomaly": 49,
            "n_normal": 68,
            "max_duration_s": 49.9,
            "seed": 20261009,
        },
        "window": {"length_s": 2.0, "stride_s": 2.0, "frames": 8},
        "input_form": {"name": "frames"},
        "memory_cap_gb": 24,
        "encoders": {
            "siglip2-so400m": "siglip2-so400m-patch14-384",
            "dinov3-vitl": "dinov3-vitl16",
        },
    }


def write_config(folder: Path, data: dict | None = None) -> Path:
    path = folder / "encoder_probe.yaml"
    payload = data if data is not None else config_data()
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def fake_plan(_config: EncoderConfig | None = None) -> dict[str, list[WindowRow]]:
    train_items = [video("a.mp4", "Abuse", 6.0, [(25, 44)]), video("n.mp4", "Normal", 6.0)]
    test_items = [video("t.mp4", "Abuse", 6.0, [(25, 44)])]
    return plan_sets(train_items, test_items, WINDOW)


def fake_frames(_row: WindowRow) -> list:
    return [None] * 8


class FakeAdapter:
    def __init__(self, name: str, with_head: bool = False) -> None:
        self.name = name
        self.kinds = ("patch_mean",)
        self.head_kinds = ("pooler_mean",) if with_head else ()
        self.n_states = STATES
        self.calls = 0

    def encode(self, frames: list) -> EncoderOutput:
        self.calls += 1
        kinds = {"patch_mean": torch.full((STATES, DIMS), float(len(frames)), dtype=torch.float32)}
        head = {}
        if self.head_kinds:
            head = {"pooler_mean": torch.ones((1, HEAD_DIMS), dtype=torch.float32)}
        return EncoderOutput(kinds=kinds, head=head)


def test_load_encoder_config_resolves_assets_and_checks_sections(tmp_path):
    path = write_config(tmp_path)
    config = load_encoder_config(path, resolver=resolver)
    assert config.name == "encoder_probe_ucf_short_v1_f8"
    assert config.train["path"] == Path("/assets/ucf-crime")
    assert config.test["path"] == Path("/assets/ucf-crime")
    assert config.window == WINDOW
    assert config.memory_cap_gb == 24
    assert config.encoder_paths == {
        "siglip2-so400m": Path("/assets/siglip2-so400m-patch14-384"),
        "dinov3-vitl": Path("/assets/dinov3-vitl16"),
    }
    missing_data = config_data()
    del missing_data["encoders"]
    bad = tmp_path / "missing.yaml"
    bad.write_text(yaml.safe_dump(missing_data))
    with pytest.raises(ConfigError, match="encoders"):
        load_encoder_config(bad, resolver=resolver)


def test_load_encoder_config_rejects_unknown_encoder(tmp_path):
    data = config_data()
    data["encoders"]["unknown-vit"] = "unknown-asset"
    bad = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="unknown-vit"):
        load_encoder_config(bad, resolver=resolver)


def test_run_encoder_extract_writes_main_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("S1_FEATURES", str(tmp_path / "features"))
    monkeypatch.setattr("s1stream.runner.free_memory_gb", lambda: 50.0)
    path = write_config(tmp_path)
    adapter = FakeAdapter("dinov3-vitl", with_head=False)
    out = run_encoder_extract(path, "dinov3-vitl", "r1",
                              loader=lambda name, asset, device: adapter,
                              plan=fake_plan, frames=fake_frames, resolver=resolver)
    assert out == tmp_path / "features" / "encoder_probe_ucf_short_v1_f8" / "dinov3-vitl" / "r1"
    assert (out / "config.yaml").exists()
    for set_name in SETS:
        arr = np.load(out / set_name / "patch_mean.npy")
        assert arr.dtype == np.float32 and arr.shape == (3, STATES, DIMS)
        assert np.all(arr == 8.0)
        meta = [json.loads(line) for line in (out / set_name / "meta.jsonl").read_text().splitlines()]
        assert len(meta) == 3
    run = json.loads((out / "run.json").read_text())
    assert run["name"] == "encoder_probe_ucf_short_v1_f8"
    assert run["encoder"] == "dinov3-vitl" and run["folder"] == "main"
    assert run["kinds"] == ["patch_mean"]
    assert run["n_states"] == STATES and run["dims"] == DIMS
    assert run["windows"] == {"train_anomaly": 3, "train_normal": 3, "test": 3}
    assert not (tmp_path / "features" / "encoder_probe_ucf_short_v1_f8" / "dinov3-vitl-head").exists()


def test_run_encoder_extract_writes_head_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("S1_FEATURES", str(tmp_path / "features"))
    monkeypatch.setattr("s1stream.runner.free_memory_gb", lambda: 50.0)
    path = write_config(tmp_path)
    adapter = FakeAdapter("siglip2-so400m", with_head=True)
    out = run_encoder_extract(path, "siglip2-so400m", "r1",
                              loader=lambda name, asset, device: adapter,
                              plan=fake_plan, frames=fake_frames, resolver=resolver)
    head_out = tmp_path / "features" / "encoder_probe_ucf_short_v1_f8" / "siglip2-so400m-head" / "r1"
    assert (head_out / "config.yaml").exists()
    for set_name in SETS:
        arr = np.load(head_out / set_name / "pooler_mean.npy")
        assert arr.dtype == np.float32 and arr.shape == (3, 1, HEAD_DIMS)
        assert np.all(arr == 1.0)
        assert (head_out / set_name / "meta.jsonl").read_text() == (out / set_name / "meta.jsonl").read_text()
    head_run = json.loads((head_out / "run.json").read_text())
    assert head_run["encoder"] == "siglip2-so400m" and head_run["folder"] == "head"
    assert head_run["kinds"] == ["pooler_mean"]
    assert head_run["n_states"] == 1 and head_run["dims"] == HEAD_DIMS


def test_existing_run_json_raises_before_windows_run(tmp_path, monkeypatch):
    monkeypatch.setenv("S1_FEATURES", str(tmp_path / "features"))
    monkeypatch.setattr("s1stream.runner.free_memory_gb", lambda: 50.0)
    path = write_config(tmp_path)
    main_dir = tmp_path / "features" / "encoder_probe_ucf_short_v1_f8" / "siglip2-so400m" / "r1"
    main_dir.mkdir(parents=True)
    (main_dir / "run.json").write_text("{}")
    adapter = FakeAdapter("siglip2-so400m", with_head=True)
    with pytest.raises(FileExistsError):
        run_encoder_extract(path, "siglip2-so400m", "r1",
                            loader=lambda name, asset, device: adapter,
                            plan=fake_plan, frames=fake_frames, resolver=resolver)
    assert adapter.calls == 0

    head_dir = tmp_path / "features" / "encoder_probe_ucf_short_v1_f8" / "siglip2-so400m-head" / "r2"
    head_dir.mkdir(parents=True)
    (head_dir / "run.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run_encoder_extract(path, "siglip2-so400m", "r2",
                            loader=lambda name, asset, device: adapter,
                            plan=fake_plan, frames=fake_frames, resolver=resolver)
    assert adapter.calls == 0


def test_low_free_memory_raises_before_loader(tmp_path, monkeypatch):
    monkeypatch.setenv("S1_FEATURES", str(tmp_path / "features"))
    monkeypatch.setattr("s1stream.runner.free_memory_gb", lambda: 1.0)
    path = write_config(tmp_path)
    loaded = []
    with pytest.raises(RuntimeError) as err:
        run_encoder_extract(path, "dinov3-vitl", "r1",
                            loader=lambda name, asset, device: loaded.append(name),
                            plan=fake_plan, frames=fake_frames, resolver=resolver)
    if isinstance(err.value, NotImplementedError):
        raise err.value
    assert loaded == []


def test_limit_windows_keeps_first_two_per_set(tmp_path, monkeypatch):
    monkeypatch.setenv("S1_FEATURES", str(tmp_path / "features"))
    monkeypatch.setattr("s1stream.runner.free_memory_gb", lambda: 50.0)
    path = write_config(tmp_path)
    adapter = FakeAdapter("dinov3-vitl")
    out = run_encoder_extract(path, "dinov3-vitl", "r1", limit_windows=2,
                              loader=lambda name, asset, device: adapter,
                              plan=fake_plan, frames=fake_frames, resolver=resolver)
    assert adapter.calls == 6
    for set_name in SETS:
        arr = np.load(out / set_name / "patch_mean.npy")
        assert arr.shape == (2, STATES, DIMS)
        meta = [json.loads(line) for line in (out / set_name / "meta.jsonl").read_text().splitlines()]
        assert [row["window"] for row in meta] == [0, 1]
    run = json.loads((out / "run.json").read_text())
    assert run["windows"] == {"train_anomaly": 2, "train_normal": 2, "test": 2}
    assert run["limit_windows"] == 2


def test_meta_rows_have_probe_extract_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("S1_FEATURES", str(tmp_path / "features"))
    monkeypatch.setattr("s1stream.runner.free_memory_gb", lambda: 50.0)
    row = fake_plan()["test"][1]
    entry = window_meta(1, row, 0.25)
    expected_keys = {"i", "video", "category", "window", "start_s", "end_s", "label", "holmes", "p_yes", "seconds"}
    assert set(entry) == expected_keys
    assert (entry["i"], entry["video"], entry["category"], entry["window"]) == (1, "t.mp4", "Abuse", 1)
    assert (entry["start_s"], entry["end_s"], entry["label"], entry["holmes"]) == (2.0, 4.0, 1, None)
    assert entry["p_yes"] is None and entry["seconds"] == 0.25

    path = write_config(tmp_path)
    adapter = FakeAdapter("dinov3-vitl")
    out = run_encoder_extract(path, "dinov3-vitl", "r1", limit_windows=2,
                              loader=lambda name, asset, device: adapter,
                              plan=fake_plan, frames=fake_frames, resolver=resolver)
    meta = [json.loads(line) for line in (out / "test" / "meta.jsonl").read_text().splitlines()]
    assert len(meta) == 2
    assert set(meta[1]) == expected_keys
    assert meta[1]["p_yes"] is None and isinstance(meta[1]["seconds"], float)


def test_cli_encoder_extract(tmp_path, monkeypatch, capsys):
    calls = []

    def fake(config_path, encoder, run_name, limit_windows=None):
        calls.append((config_path, encoder, run_name, limit_windows))
        return tmp_path / "out"

    monkeypatch.setattr("s1stream.encoder.extract.run_encoder_extract", fake)
    code = cli.main(["encoder-extract", "cfg.yaml", "--encoder", "dinov3-vitl", "--run", "r1", "--limit-windows", "5"])
    assert code == 0
    assert calls == [(Path("cfg.yaml"), "dinov3-vitl", "r1", 5)]
    assert capsys.readouterr().out.strip() == str(tmp_path / "out")

    def bad(config_path, encoder, run_name, limit_windows=None):
        raise ConfigError("bad config")

    monkeypatch.setattr("s1stream.encoder.extract.run_encoder_extract", bad)
    assert cli.main(["encoder-extract", "cfg.yaml", "--encoder", "dinov3-vitl", "--run", "r1"]) == 1
