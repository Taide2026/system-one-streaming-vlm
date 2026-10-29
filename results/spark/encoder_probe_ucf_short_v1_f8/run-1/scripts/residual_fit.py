"""Probe fit on [feature, residual]: each window feature concatenated with (feature - causal past mean), per state.

Question: centering (centered_fit.py) removes the scene, but the head then loses the scene and the event type, and
the causal versions are unstable. Here the head gets both: x_t and r_t = x_t - reference_t, so a linear head can use
w1 * x_t + w2 * r_t and learn the mix itself. reference_t comes from centered_fit.reference (modes past<N>, pastall,
full); the first window of a video has no earlier window, so its residual is the zero vector.
Usage: uv run python <this file> <feature folder> [kind] [mode]   (default patch_mean pastall)
Writes the features to $S1_FEATURES/encoder_probe_ucf_short_v1_f8_residual_<mode>/<folder>/run-1/ (kind only, last
axis 2 * hidden) and the fit to run-1/analysis-residual/fit-<folder>-<mode>/, then prints the report of centered_fit.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from centered_fit import reference, report
from s1stream.probe.fit import SETS, fit_condition

RUN = Path(__file__).resolve().parents[1]
FEATURES = Path(os.environ["S1_FEATURES"])


def with_residual(block: np.ndarray, starts: np.ndarray, mode: str) -> np.ndarray:
    """[windows, states, 2 * hidden]: the feature, then the feature minus its reference (one video)."""
    return np.concatenate([block, block - reference(block, starts, mode)], axis=-1)


def write_features(source: Path, target: Path, kind: str, mode: str) -> None:
    """Write target/<set>/<kind>.npy (float32, with_residual per video) and copy meta.jsonl; run.json kinds [kind].
    Rows of one video are contiguous in meta.jsonl."""
    for name in SETS:
        (target / name).mkdir(parents=True, exist_ok=True)
        shutil.copy(source / name / "meta.jsonl", target / name / "meta.jsonl")
        meta = [json.loads(line) for line in (source / name / "meta.jsonl").read_text().splitlines()]
        videos = np.array([row["video"] for row in meta])
        starts = np.array([row["start_s"] for row in meta])
        x = np.load(source / name / f"{kind}.npy", mmap_mode="r")
        shape = (*x.shape[:-1], 2 * x.shape[-1])
        out = np.lib.format.open_memmap(target / name / f"{kind}.npy", mode="w+", dtype=np.float32, shape=shape)
        for video in dict.fromkeys(videos):
            rows = np.flatnonzero(videos == video)
            first, last = rows[0], rows[-1] + 1
            out[first:last] = with_residual(np.asarray(x[first:last], dtype=np.float32), starts[first:last], mode)
        out.flush()
    run = json.loads((source / "run.json").read_text())
    run.update(kinds=[kind], residual_from=str(source), residual_mode=mode)
    (target / "run.json").write_text(json.dumps(run, indent=1) + "\n")


def main(folder: str, kind: str = "patch_mean", mode: str = "pastall") -> None:
    source = FEATURES / "encoder_probe_ucf_short_v1_f8" / folder / "run-1"
    target = FEATURES / f"encoder_probe_ucf_short_v1_f8_residual_{mode}" / folder / "run-1"
    fit_dir = RUN / "analysis-residual" / f"fit-{folder}-{mode}"
    if not (target / "run.json").exists():
        write_features(source, target, kind, mode)
    if not (fit_dir / "fit.json").exists():
        torch.cuda.set_per_process_memory_fraction(0.2)
        fit_condition(target, fit_dir, device="cuda")
    report(fit_dir, kind)


if __name__ == "__main__":
    main(*sys.argv[1:])
