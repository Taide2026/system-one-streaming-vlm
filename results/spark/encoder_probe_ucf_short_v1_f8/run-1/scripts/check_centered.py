"""Checks for centered_fit.py (2026-10-29).

1. report() on the original fit-qwen35-vit must give the known MIL numbers (all 0.807, in_video 0.566).
2. The centered feature files must equal a direct recomputation for a few videos (full and pastall, test set).
3. Validation scores in fit.json per mode: MIL video BCE (ln 2 = 0.693 means no video-level information) and
   Holmes window AUC, at the state chosen on validation.
Usage: uv run python <this file>
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from centered_fit import report  # noqa: E402

from s1stream.encoder.compare import select_state  # noqa: E402

RUN = Path(__file__).resolve().parents[1]
FEATURES = Path(os.environ["S1_FEATURES"])
FOLDER = "vjepa2-vitl"
MODES = {"full": "", "past10": "_past10", "past20": "_past20", "pastall": "_pastall"}


def check_report() -> None:
    print("== 1. report() on the original Qwen ViT fit (expect mil all 0.807, in_video 0.566)")
    report(RUN / "fit-qwen35-vit", "patch_mean")


def expected(block: np.ndarray, starts: np.ndarray, mode: str) -> np.ndarray:
    """Direct, loop-based recomputation of the centered features of one video."""
    out = np.empty_like(block)
    for i in range(len(block)):
        if mode == "full":
            reference = block.mean(axis=0)
        else:
            past = [j for j in range(len(block)) if starts[j] < starts[i]]
            reference = block[past].mean(axis=0) if past else block[i]
        out[i] = block[i] - reference
    return out


def check_features() -> None:
    print(f"== 2. centered features of {FOLDER} vs direct recomputation (test set, 5 videos)")
    source = FEATURES / "encoder_probe_ucf_short_v1_f8" / FOLDER / "run-1" / "test"
    meta = [json.loads(line) for line in (source / "meta.jsonl").read_text().splitlines()]
    videos = np.array([row["video"] for row in meta])
    starts = np.array([row["start_s"] for row in meta])
    original = np.load(source / "patch_mean.npy", mmap_mode="r")
    for mode in ("full", "pastall"):
        centered = np.load(FEATURES / f"encoder_probe_ucf_short_v1_f8_centered{MODES[mode]}" / FOLDER / "run-1"
                           / "test" / "patch_mean.npy", mmap_mode="r")
        worst = 0.0
        for video in list(dict.fromkeys(videos))[:5]:
            rows = np.flatnonzero(videos == video)
            block = np.asarray(original[rows], dtype=np.float32)
            worst = max(worst, float(np.abs(np.asarray(centered[rows]) - expected(block, starts[rows], mode)).max()))
        print(f"  {mode:8s} max abs difference {worst:.2e}")


def check_validation() -> None:
    print("== 3. validation at the chosen state (MIL: video BCE, ln 2 = 0.693; Holmes: window AUC)")
    fits = {"original": RUN / f"fit-{FOLDER}"}
    for mode, suffix in MODES.items():
        fits[mode] = RUN / "analysis-centered" / f"fit-{FOLDER}{suffix.replace('_', '-')}"
    for name, fit_dir in fits.items():
        fit = json.loads((fit_dir / "fit.json").read_text())
        mil_state = select_state(fit, "mil", "patch_mean")
        holmes_state = select_state(fit, "holmes", "patch_mean")
        mil_bce = fit["choices"][f"mil/patch_mean/{mil_state}"]["validation"]
        holmes_auc = fit["choices"][f"holmes/patch_mean/{holmes_state}"]["validation"]
        mil_all = [fit["choices"][f"mil/patch_mean/{s}"]["validation"] for s in range(fit["n_states"])]
        print(f"  {name:8s} MIL state {mil_state:2d} BCE {mil_bce:.3f} (range over states {min(mil_all):.3f}-{max(mil_all):.3f})"
              f"   Holmes state {holmes_state:2d} AUC {holmes_auc:.3f}")


if __name__ == "__main__":
    check_report()
    check_features()
    check_validation()
