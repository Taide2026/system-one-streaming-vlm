"""Last stage of encoder probing: `s1stream encoder-compare` (encoder-probing decisions 7, 8).

Input: a results run folder with one fit folder per encoder feature folder (fit-<encoder>, fit-<encoder>-head; written
by probe-fit), and the encoder config (for the test videos). Each fit has its own kinds and state count (fit.json).

Per fit folder, kind and method, on the test windows: the frame AUC curve over the states, with N_BOOT video resamples
(seed SEED; the same resamples, drawn as in probe.compare.bootstrap, for every fit, so differences are paired).
Three states are reported:
  selected: select_state (validation only, decision 7)
  last:     the last state
  peak:     the state with the highest test frame AUC (reference only; the first one on a tie)
Difference against the baseline (BASELINE: fit-qwen35-vit, kind patch_mean, decision 8), per method: frame AUC of this
fit, kind and method at its selected state minus the baseline frame AUC at the baseline's selected state of the same
method, paired over the resamples.

Writes into the run folder:
  encoder_compare.json {"fits": [...], "methods": [...], "n_boot": N_BOOT, "seed": SEED,
                        "baseline": {"fit": "qwen35-vit", "kind": "patch_mean"},
                        "results": {"<fit>/<kind>/<method>": {
                            "n_states": S, "curve": {"value": [S], "low": [S], "high": [S]},
                            "selected": R, "last": R, "peak": R,
                            "difference": {"state": i, "baseline_state": j, "value": x, "low": x, "high": x}}}}
                       where R = {"state": i, "value": x, "low": x, "high": x}; <fit> is the folder name without "fit-".
  <method>.png         frame AUC per state, one line per fit and kind.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from s1stream.types import VideoItem

N_BOOT, SEED = 1000, 20261022
BASELINE = ("qwen35-vit", "patch_mean")


def select_state(fit: dict, method: str, kind: str) -> int:
    """The state chosen on validation from fit.json choices "<method>/<kind>/<state>" "validation":
    mil the lowest value (video BCE), holmes and knn the highest (AUC). A tie goes to the lower state."""
    raise NotImplementedError


def run_encoder_compare(config_path: Path, run_dir: Path, items: dict[str, VideoItem] | None = None,
                        log: Callable[[str], None] = print) -> Path:
    """Read every run_dir/fit-*/ (fit.json, test_scores.npy, test_meta.jsonl) and write encoder_compare.json and the PNG
    files (matplotlib, Agg backend). Return run_dir / "encoder_compare.json".
    Raise FileExistsError when encoder_compare.json exists; ValueError when fit-qwen35-vit is missing or has no
    patch_mean kind; ValueError when the probe.compare.WINDOW_KEYS of the test_meta.jsonl rows differ between two
    fit folders. items None: build the test items from the config (encoder.extract.load_encoder_config, registry
    DATASETS); given items: the config is not read."""
    raise NotImplementedError
