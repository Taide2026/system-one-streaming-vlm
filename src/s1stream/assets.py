"""External assets (proposal decision 15): models and datasets named in assets.yaml.

Asset <name> lives at asset_root() / <name>. asset_root() is $S1_ASSETS, or ./assets when S1_ASSETS is not set.
fetch downloads an asset, runs its prepare step, and checks its hashes. check only checks.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

LOCK_FILE = Path(__file__).resolve().parents[2] / "assets.yaml"
KINDS = ("hf_model", "archive")
REQUIRED = {"hf_model": ("kind", "source", "revision", "sha256"),
            "archive": ("kind", "source", "size", "sha256")}


class AssetError(Exception):
    """An asset problem. The message names the asset, and the file or folder when there is one."""


@dataclass(frozen=True)
class Asset:
    name: str
    kind: str                  # one of KINDS
    source: str                # hf_model: repo id; archive: download URL
    sha256: dict[str, str]     # relative path -> hex digest
    revision: str | None = None
    size: int | None = None    # archive: zip size in bytes
    prepare: str | None = None  # archive: key of PREPARES


Downloader = Callable[[Asset, Path], None]  # (asset, target folder) -> files in the folder


def load_lock(path: Path) -> dict[str, Asset]:
    """Read the lock file. Raise AssetError naming the entry when its kind is unknown or a field of REQUIRED is missing."""
    raise NotImplementedError


def asset_root() -> Path:
    """$S1_ASSETS when set, else Path.cwd() / "assets". Read the environment at call time."""
    raise NotImplementedError


def resolve(name: str, lock: dict[str, Asset] | None = None) -> Path:
    """Return asset_root() / name.

    lock defaults to load_lock(LOCK_FILE), with LOCK_FILE read at call time (tests replace it).
    Raise AssetError when name is not in lock, or when the folder does not exist (the message names both).
    """
    raise NotImplementedError


def check(asset: Asset, folder: Path) -> None:
    """Raise AssetError naming the file when a file of asset.sha256 is missing or has another SHA-256."""
    raise NotImplementedError


def hf_download(asset: Asset, folder: Path) -> None:
    """huggingface_hub.snapshot_download(repo_id=asset.source, revision=asset.revision, local_dir=folder)."""
    raise NotImplementedError


def archive_download(asset: Asset, folder: Path) -> None:
    """Download asset.source to folder / "download.zip", raise AssetError when the size is not asset.size,
    unzip it into folder, then delete the zip."""
    raise NotImplementedError


def prepare_ucf_crime(folder: Path) -> None:
    """Turn the unzipped UCF-Crime archive in folder into the layout of the asset README.

    Input (the outer zip's files): Anomaly-Videos-Part-*.zip, Training-Normal-Videos-Part-*.zip,
    Testing_Normal_Videos.zip, Normal_Videos_for_Event_Recognition.zip, UCF_Crimes-Train-Test-Split.zip,
    Anomaly_Train.txt, Temporal_Anomaly_Annotation_for_Testing_Videos.txt, ReadMe-Anomaly-Detection.txt.
    Output:
      videos/<Class>/<file>                          from Anomaly-Videos-Part-*/<Class>/
      videos/Training_Normal_Videos_Anomaly/<file>    from Training-Normal-Videos-Part-*/
      videos/Testing_Normal_Videos_Anomaly/<file>
      videos/Normal_Videos_event/<file>               from Normal_Videos_for_Event_Recognition/
      annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
      splits/Anomaly_Train.txt (the top-level file), splits/Anomaly_Test.txt, splits/Action_Regnition_splits/
      ReadMe-Anomaly-Detection.txt
    The empty Anomaly_Detection_splits/Anomaly_Train.txt is deleted. Each inner zip is deleted after it is extracted.
    """
    raise NotImplementedError


DOWNLOADERS: dict[str, Downloader] = {"hf_model": hf_download, "archive": archive_download}
PREPARES: dict[str, Callable[[Path], None]] = {"ucf_crime": prepare_ucf_crime}


def fetch(asset: Asset, folder: Path, download: Downloader | None = None) -> None:
    """Make folder hold a checked copy of asset.

    1. When folder exists and check passes, return without downloading.
    2. Else call download (default DOWNLOADERS[asset.kind]) with (asset, folder), creating folder first.
    3. When asset.prepare is set, call PREPARES[asset.prepare](folder).
    4. Call check; its AssetError propagates.
    """
    raise NotImplementedError
