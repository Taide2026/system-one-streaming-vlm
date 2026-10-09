"""Unit assets: lock file, resolve, check, fetch, ucf_crime prepare step, config asset resolution."""
import hashlib
import zipfile
from pathlib import Path

import pytest
import yaml

import s1stream.assets as assets
from s1stream.assets import Asset, AssetError, check, fetch, load_lock, prepare_ucf_crime, resolve
from s1stream.config import ConfigError, load_config

HELLO = b"hello\n"
HELLO_SHA = hashlib.sha256(HELLO).hexdigest()

LOCK = {
    "jev-omni": {"kind": "hf_model", "source": "org/Jev-Omni", "revision": "abc123", "sha256": {"head.pt": HELLO_SHA}},
    "ucf-crime": {"kind": "archive", "source": "https://example.com/ucf.zip", "size": 10, "prepare": "ucf_crime",
                  "sha256": {"README.md": HELLO_SHA}},
}


def write_lock(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "assets.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def model_asset() -> Asset:
    return Asset(name="jev-omni", kind="hf_model", source="org/Jev-Omni", revision="abc123", sha256={"head.pt": HELLO_SHA})


def test_load_lock_reads_both_kinds(tmp_path):
    lock = load_lock(write_lock(tmp_path, LOCK))
    assert lock["jev-omni"] == model_asset()
    ucf = lock["ucf-crime"]
    assert (ucf.kind, ucf.source, ucf.size, ucf.prepare) == ("archive", "https://example.com/ucf.zip", 10, "ucf_crime")


def test_load_lock_missing_field_names_entry(tmp_path):
    data = {"jev-omni": {k: v for k, v in LOCK["jev-omni"].items() if k != "revision"}}
    with pytest.raises(AssetError) as err:
        load_lock(write_lock(tmp_path, data))
    assert "jev-omni" in str(err.value)


def test_resolve_uses_s1_assets(tmp_path, monkeypatch):
    (tmp_path / "jev-omni").mkdir()
    monkeypatch.setenv("S1_ASSETS", str(tmp_path))
    assert resolve("jev-omni", {"jev-omni": model_asset()}) == tmp_path / "jev-omni"


def test_resolve_defaults_to_dot_assets(tmp_path, monkeypatch):
    (tmp_path / "assets" / "jev-omni").mkdir(parents=True)
    monkeypatch.delenv("S1_ASSETS", raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve("jev-omni", {"jev-omni": model_asset()}) == tmp_path / "assets" / "jev-omni"


def test_resolve_missing_folder_names_asset_and_path(tmp_path, monkeypatch):
    monkeypatch.setenv("S1_ASSETS", str(tmp_path))
    with pytest.raises(AssetError) as err:
        resolve("jev-omni", {"jev-omni": model_asset()})
    assert "jev-omni" in str(err.value)
    assert str(tmp_path) in str(err.value)


def test_check_passes_on_matching_hashes(tmp_path):
    (tmp_path / "head.pt").write_bytes(HELLO)
    check(model_asset(), tmp_path)


def test_check_names_wrong_and_missing_file(tmp_path):
    (tmp_path / "head.pt").write_bytes(b"other\n")
    with pytest.raises(AssetError) as err:
        check(model_asset(), tmp_path)
    assert "head.pt" in str(err.value)
    (tmp_path / "head.pt").unlink()
    with pytest.raises(AssetError) as err:
        check(model_asset(), tmp_path)
    assert "head.pt" in str(err.value)


def test_fetch_calls_downloader_with_repo_revision_and_folder_then_checks(tmp_path):
    calls = []

    def fake_download(asset: Asset, folder: Path) -> None:
        calls.append((asset.source, asset.revision, folder))
        (folder / "head.pt").write_bytes(b"wrong\n")

    target = tmp_path / "jev-omni"
    with pytest.raises(AssetError):  # the downloaded file has the wrong hash, so check must run and fail
        fetch(model_asset(), target, download=fake_download)
    assert calls == [("org/Jev-Omni", "abc123", target)]


def test_fetch_skips_download_when_check_passes(tmp_path):
    target = tmp_path / "jev-omni"
    target.mkdir()
    (target / "head.pt").write_bytes(HELLO)

    def fail_download(asset: Asset, folder: Path) -> None:
        raise AssertionError("download must not run")

    fetch(model_asset(), target, download=fail_download)


def make_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def test_prepare_ucf_crime_builds_layout(tmp_path):
    make_zip(tmp_path / "Anomaly-Videos-Part-1.zip", {"Anomaly-Videos-Part-1/Abuse/Abuse001_x264.mp4": b"a"})
    make_zip(tmp_path / "Anomaly-Videos-Part-2.zip", {"Anomaly-Videos-Part-2/Burglary/Burglary001_x264.mp4": b"b"})
    make_zip(tmp_path / "Training-Normal-Videos-Part-1.zip",
             {"Training-Normal-Videos-Part-1/Normal_Videos001_x264.mp4": b"c"})
    make_zip(tmp_path / "Training-Normal-Videos-Part-2.zip",
             {"Training-Normal-Videos-Part-2/Normal_Videos483_x264.mp4": b"d"})
    make_zip(tmp_path / "Testing_Normal_Videos.zip", {"Testing_Normal_Videos_Anomaly/Normal_Videos_003_x264.mp4": b"e"})
    make_zip(tmp_path / "Normal_Videos_for_Event_Recognition.zip",
             {"Normal_Videos_for_Event_Recognition/Normal_Videos_015_x264.mp4": b"f"})
    make_zip(tmp_path / "UCF_Crimes-Train-Test-Split.zip", {
        "Anomaly_Detection_splits/Anomaly_Train.txt": b"",
        "Anomaly_Detection_splits/Anomaly_Test.txt": b"Abuse/Abuse001_x264.mp4\n",
        "Action_Regnition_splits/ClassIDs.txt": b"Abuse 1\n",
    })
    (tmp_path / "Anomaly_Train.txt").write_text("Abuse/Abuse001_x264.mp4\n")
    (tmp_path / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").write_text("x\n")
    (tmp_path / "ReadMe-Anomaly-Detection.txt").write_text("readme\n")

    prepare_ucf_crime(tmp_path)

    files = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert files == sorted([
        "ReadMe-Anomaly-Detection.txt",
        "annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
        "splits/Anomaly_Train.txt",
        "splits/Anomaly_Test.txt",
        "splits/Action_Regnition_splits/ClassIDs.txt",
        "videos/Abuse/Abuse001_x264.mp4",
        "videos/Burglary/Burglary001_x264.mp4",
        "videos/Training_Normal_Videos_Anomaly/Normal_Videos001_x264.mp4",
        "videos/Training_Normal_Videos_Anomaly/Normal_Videos483_x264.mp4",
        "videos/Testing_Normal_Videos_Anomaly/Normal_Videos_003_x264.mp4",
        "videos/Normal_Videos_event/Normal_Videos_015_x264.mp4",
    ])
    assert (tmp_path / "splits" / "Anomaly_Train.txt").read_text() == "Abuse/Abuse001_x264.mp4\n"


def test_load_config_resolves_asset_and_rejects_unknown_asset(tmp_path, monkeypatch):
    (tmp_path / "jev-omni").mkdir()
    monkeypatch.setenv("S1_ASSETS", str(tmp_path))
    monkeypatch.setattr(assets, "LOCK_FILE", write_lock(tmp_path, LOCK))
    config = {
        "name": "demo",
        "dataset": {"name": "ucf_crime"},
        "window": {"length_s": 2.0, "stride_s": 2.0, "frames": 8},
        "question": {"state": "s", "text": "q", "options": ["Yes", "No"], "positive": "Yes"},
        "input_form": {"name": "frames"},
        "backbone": {"name": "jev_omni", "asset": "jev-omni"},
        "answer_head": {"name": "option_prob"},
        "decision_rules": [{"name": "threshold"}],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    cfg = load_config(path)
    assert cfg.backbone["path"] == tmp_path / "jev-omni"
    assert "path" not in cfg.raw["backbone"]
    config["backbone"]["asset"] = "not-in-lock"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ConfigError) as err:
        load_config(path)
    assert "not-in-lock" in str(err.value)
