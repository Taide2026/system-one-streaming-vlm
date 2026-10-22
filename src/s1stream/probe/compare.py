"""Last stage of probing: `s1stream probe-compare` (finetune-layer-probing decisions 9, 11, 12).

Input: a results run folder with one fit folder per condition (fit-A, fit-B, fit-C, written by probe-fit), and the
probe config (for the test videos: fps, frame count and annotated events).

Metrics per condition, method, token kind and state, on the test windows:
  frame_auc (main), frame_ap: s1stream.metrics conventions (frame ranges from windows.window_frame_range);
  window_auc: s1stream.metrics.auc on the window scores and window labels.
P(Yes) of every condition (meta "p_yes") is scored the same way, as a reference line (decision 9).

Bootstrap: N_BOOT resamples of the test videos with replacement (numpy default_rng(SEED)); every condition, method,
kind and state uses the same resamples, so differences are paired. A resample whose videos have no abnormal frame
or no normal frame gives NaN for that metric; intervals are the 2.5 and 97.5 percentiles over the finite values
(np.nanpercentile). The point value is the metric on the original test set.

Layer summary (decision 12) of a frame AUC curve over the states: layer_summary. Its onset state, peak state and
peak value are also computed on every resample, which gives their intervals and the paired intervals of their
differences between conditions.

Writes into the run folder:
  compare.json  {"conditions": [...], "methods": [...], "kinds": [...], "n_states": S, "n_boot": N_BOOT,
                 "seed": SEED, "fits": {<condition>: <fit dir>},
                 "metrics": {"<cond>/<method>/<kind>/<metric>": {"value": [S], "low": [S], "high": [S]}},
                 "p_yes": {"<cond>/<metric>": {"value": x, "low": x, "high": x}},
                 "differences": {"<pair>/<method>/<kind>/<metric>": {"value": [S], "low": [S], "high": [S]}},
                 "summary": {"<cond>/<method>/<kind>": {"onset": I, "peak": I, "peak_value": I}},
                 "summary_differences": {"<pair>/<method>/<kind>": {"onset": I, "peak": I, "peak_value": I}},
                 "cka": {"<pair>/<kind>": [S]}}
                where I = {"value": x, "low": x, "high": x}; pair is one of PAIRS, e.g. "C-B" = C minus B.
  <method>_<kind>.png  frame AUC per state, one line per condition with its 95 % band, P(Yes) as dashed lines.
  cka.png              linear CKA per state for every pair and kind.
Conditions missing from the run folder are left out, together with their pairs.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from s1stream.types import VideoItem

METRICS = ("frame_auc", "frame_ap", "window_auc")
PAIRS = (("A", "B"), ("C", "B"), ("C", "A"))
N_BOOT, SEED = 1000, 20261022


def frame_metric(scores: Sequence[float], meta: list[dict], items: dict[str, VideoItem], metric: str) -> float | None:
    """One metric of METRICS for one score per test window. meta: the test_meta.jsonl rows (video, start_s, end_s,
    label), in the order of scores; items: test VideoItem by name (fps, n_frames, events). frame_auc and frame_ap
    use metrics.frame_auc / metrics.frame_ap with window_frame_range(start_s, end_s, fps); window_auc uses
    metrics.auc with meta "label"."""
    raise NotImplementedError


def layer_summary(curve: Sequence[float]) -> dict[str, float | int]:
    """{"onset": first state with value >= 0.5 + 0.9 * (max - 0.5), "peak": first state with the max value,
    "peak_value": max}. NaN values are ignored; all-NaN gives NaN for all three."""
    raise NotImplementedError


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA of two (n, p) and (n, q) matrices: columns centered, ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F),
    computed in float64."""
    raise NotImplementedError


def bootstrap(scores: dict[str, np.ndarray], p_yes: dict[str, np.ndarray], meta: list[dict],
              items: dict[str, VideoItem], n: int = N_BOOT, seed: int = SEED) -> dict:
    """The "metrics", "p_yes", "differences", "summary" and "summary_differences" parts of compare.json (see the
    module docstring). scores: per condition an array (methods, kinds, S, n_test) as in test_scores.npy; p_yes: per
    condition an array (n_test,); method and kind names in the keys come from probe.fit.METHODS and KINDS.
    The frame metrics must equal frame_metric exactly; precompute per video the frame counts of each window and add
    them per resample instead of building frame arrays again."""
    raise NotImplementedError


def run_compare(config_path: Path, run_dir: Path, items: dict[str, VideoItem] | None = None,
                log: Callable[[str], None] = print) -> Path:
    """Read run_dir/fit-<condition>/ (fit.json, test_scores.npy, test_meta.jsonl) for each condition present;
    raise ValueError when the test_meta.jsonl rows of two conditions differ, FileNotFoundError without any fit
    folder and FileExistsError when run_dir/compare.json exists. items None: build the test items from the config
    (probe.extract.load_probe_config, registry DATASETS); given items: the config is not read. CKA reads the test features of each fit (fit.json
    "features" / "test" / "<kind>.npy") one state at a time (mmap). Write compare.json and the PNG files with
    matplotlib (Agg backend). Return run_dir / "compare.json"."""
    raise NotImplementedError
