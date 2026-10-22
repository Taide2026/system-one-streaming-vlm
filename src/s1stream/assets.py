"""External assets (proposal decision 15): models and datasets named in assets.yaml.

Asset <name> lives at asset_root() / <name>. asset_root() is $S1_ASSETS, or ./assets when S1_ASSETS is not set.
fetch downloads an asset, runs its prepare step, and checks its hashes. check only checks.
"""
from __future__ import annotations

import hashlib
import http.cookiejar
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

LOCK_FILE = Path(__file__).resolve().parents[2] / "assets.yaml"
KINDS = ("hf_model", "archive", "url_file")
REQUIRED = {"hf_model": ("kind", "source", "revision", "sha256"),
            "archive": ("kind", "source", "size", "sha256"),
            "url_file": ("kind", "source", "file", "sha256")}


class AssetError(Exception):
    """An asset problem. The message names the asset, and the file or folder when there is one."""


@dataclass(frozen=True)
class Asset:
    name: str
    kind: str
    source: str
    sha256: dict[str, str]
    revision: str | None = None
    size: int | None = None
    prepare: str | None = None
    file: str | None = None  # url_file: the name the downloaded file gets in the asset folder


Downloader = Callable[[Asset, Path], None]


def load_lock(path: Path) -> dict[str, Asset]:
    """Read the lock file and validate its entries."""
    import yaml

    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise AssetError(f"Could not read asset lock {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AssetError(f"Invalid asset lock: {path}")
    result = {}
    for name, values in data.items():
        if not isinstance(values, dict) or values.get("kind") not in KINDS:
            raise AssetError(f"Asset {name!r} has an unknown kind")
        missing = [key for key in REQUIRED[values["kind"]] if key not in values]
        if missing:
            raise AssetError(f"Asset {name!r} is missing required field {missing[0]!r}")
        result[name] = Asset(name=name, kind=values["kind"], source=values["source"],
                             sha256=values["sha256"], revision=values.get("revision"),
                             size=values.get("size"), prepare=values.get("prepare"),
                             file=values.get("file"))
    return result


def asset_root() -> Path:
    """Return the configured asset root."""
    value = os.environ.get("S1_ASSETS")
    return Path(value) if value else Path.cwd() / "assets"


def resolve(name: str, lock: dict[str, Asset] | None = None) -> Path:
    """Return the existing folder for a locked asset."""
    if lock is None:
        lock = load_lock(LOCK_FILE)
    if name not in lock:
        raise AssetError(f"Asset {name!r} is not in the lock file")
    folder = asset_root() / name
    if not folder.is_dir():
        raise AssetError(f"Asset {name!r} folder does not exist: {folder}")
    return folder


def check(asset: Asset, folder: Path) -> None:
    """Check every locked file hash."""
    for relative, expected in asset.sha256.items():
        path = folder / relative
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(16 * 1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise AssetError(f"Asset {asset.name!r} file {relative!r} is missing or unreadable") from exc
        if digest.hexdigest() != expected:
            raise AssetError(f"Asset {asset.name!r} file {relative!r} has an incorrect SHA-256")


def hf_download(asset: Asset, folder: Path) -> None:
    """Download a pinned Hugging Face repository snapshot."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=asset.source, revision=asset.revision, local_dir=folder)


def archive_download(asset: Asset, folder: Path) -> None:
    """Download, size-check, and extract an archive. Cookies are kept across redirects: a SharePoint share link sets
    one before it serves the file."""
    path = folder / "download.zip"
    size = 0
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    with opener.open(asset.source) as response, path.open("wb") as output:
        while chunk := response.read(16 * 1024 * 1024):
            output.write(chunk)
            size += len(chunk)
    if size != asset.size:
        raise AssetError(f"Asset {asset.name!r} archive {path} has size {size}, expected {asset.size}")
    with zipfile.ZipFile(path) as archive:
        archive.extractall(folder)
    path.unlink()


def _move_contents(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if destination.exists() and item.is_dir():
            _move_contents(item, destination)
        else:
            shutil.move(str(item), str(destination))
    source.rmdir()


def prepare_ucf_crime(folder: Path) -> None:
    """Convert the UCF-Crime archive layout to the asset layout."""
    videos = folder / "videos"
    for archive_path in sorted(folder.glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(folder)
        archive_path.unlink()
        name = archive_path.stem
        extracted = folder / name
        if name.startswith("Anomaly-Videos-Part-"):
            _move_contents(extracted, videos)
        elif name.startswith("Training-Normal-Videos-Part-"):
            _move_contents(extracted, videos / "Training_Normal_Videos_Anomaly")
        elif name == "Testing_Normal_Videos":
            source = folder / "Testing_Normal_Videos_Anomaly"
            _move_contents(source, videos / "Testing_Normal_Videos_Anomaly")
            if extracted.exists():
                extracted.rmdir()

        elif name == "Normal_Videos_for_Event_Recognition":
            _move_contents(extracted, videos / "Normal_Videos_event")
        elif name == "UCF_Crimes-Train-Test-Split":
            splits = folder / "splits"
            anomaly = folder / "Anomaly_Detection_splits"
            action = folder / "Action_Regnition_splits"
            anomaly_train = anomaly / "Anomaly_Train.txt"
            if anomaly_train.exists():
                anomaly_train.unlink()
            _move_contents(anomaly, splits)
            _move_contents(action, splits / "Action_Regnition_splits")
            if extracted.exists():
                extracted.rmdir()
    annotation = folder / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    if annotation.exists():
        destination = folder / "annotations" / annotation.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        annotation.rename(destination)
    top_train = folder / "Anomaly_Train.txt"
    if top_train.exists():
        destination = folder / "splits" / top_train.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        top_train.rename(destination)
    for empty in (folder / "Anomaly_Detection_splits", folder / "Action_Regnition_splits"):
        if empty.exists():
            empty.rmdir()


XD_VIOLENCE_ANNOTATIONS = "https://roc-ng.github.io/XD-Violence/images/annotations.txt"


def prepare_xd_violence(folder: Path) -> None:
    """The XD-Violence test archive already holds videos/ (the 800 test videos); add the test annotations from the
    project page as annotations/annotations.txt (its hash is locked)."""
    target = folder / "annotations" / "annotations.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(XD_VIOLENCE_ANNOTATIONS) as response:
        target.write_bytes(response.read())


def url_download(asset: Asset, folder: Path, opener: Callable[[str], bytes] | None = None) -> None:
    """Download one file: the bytes at asset.source go to folder / asset.file. opener (url -> bytes) is for tests."""
    if opener is None:
        def opener(url: str) -> bytes:
            with urllib.request.urlopen(url) as response:
                return response.read()
    (folder / asset.file).write_bytes(opener(asset.source))


DOWNLOADERS: dict[str, Downloader] = {"hf_model": hf_download, "archive": archive_download,
                                      "url_file": url_download}
PREPARES: dict[str, Callable[[Path], None]] = {"ucf_crime": prepare_ucf_crime, "xd_violence": prepare_xd_violence}


def fetch(asset: Asset, folder: Path, download: Downloader | None = None) -> None:
    """Download, prepare, and verify an asset."""
    if folder.exists():
        try:
            check(asset, folder)
            return
        except AssetError:
            pass
    folder.mkdir(parents=True, exist_ok=True)
    downloader = download if download is not None else DOWNLOADERS[asset.kind]
    downloader(asset, folder)
    if asset.prepare is not None:
        PREPARES[asset.prepare](folder)
    check(asset, folder)
