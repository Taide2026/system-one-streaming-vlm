"""Unit assets (finetune-layer-probing decision 2): the url_file kind and the two probing assets in assets.yaml."""
import hashlib
from pathlib import Path

import pytest
import yaml

from s1stream.assets import LOCK_FILE, Asset, AssetError, fetch, load_lock, url_download

HELLO = b"hello\n"
HELLO_SHA = hashlib.sha256(HELLO).hexdigest()
ENTRY = {"kind": "url_file", "source": "https://example.com/train.json", "file": "train.json",
         "sha256": {"train.json": HELLO_SHA}}


def write_lock(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "assets.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def url_asset() -> Asset:
    return Asset(name="notes", kind="url_file", source=ENTRY["source"], sha256=ENTRY["sha256"], file="train.json")


def test_load_lock_reads_url_file(tmp_path):
    assert load_lock(write_lock(tmp_path, {"notes": ENTRY}))["notes"] == url_asset()


def test_load_lock_url_file_without_file_names_field(tmp_path):
    entry = {key: value for key, value in ENTRY.items() if key != "file"}
    with pytest.raises(AssetError) as err:
        load_lock(write_lock(tmp_path, {"notes": entry}))
    assert "file" in str(err.value)


def test_fetch_url_file_writes_and_checks(tmp_path):
    folder = tmp_path / "notes"
    fetch(url_asset(), folder, lambda asset, target: url_download(asset, target, opener=lambda url: HELLO))
    assert (folder / "train.json").read_bytes() == HELLO


def test_fetch_url_file_wrong_bytes_fails(tmp_path):
    with pytest.raises(AssetError):
        fetch(url_asset(), tmp_path / "notes",
              lambda asset, target: url_download(asset, target, opener=lambda url: b"other\n"))


def test_repo_lock_has_probing_assets():
    lock = load_lock(LOCK_FILE)
    qwen, holmes = lock["qwen3.5-9b"], lock["holmes-vau-ucf-train"]
    assert (qwen.kind, qwen.source, qwen.revision) == ("hf_model", "Qwen/Qwen3.5-9B",
                                                        "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    assert (holmes.kind, holmes.file) == ("url_file", "ucf_database_train.json")
    assert holmes.sha256 == {"ucf_database_train.json":
                             "eef3357ccb5bfe446e3b7a88d8aedd18b32b2706fab928dfecf1cd81183b0369"}
