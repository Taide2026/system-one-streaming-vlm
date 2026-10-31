"""Pixel motion of each window: does motion localize the event where the encoder features follow the scene?

Question: heads on encoder features learn the scene (event_phase_auc.py). Motion inside a 2 s window (people
fighting, running, a car hitting) is closer to the event and is near zero for a static scene. This measures motion
from the pixels, without a model, on the same 8 frames per window that the encoders saw (probe.extract.plan_sets).
Motion features per window (8 frames -> 7 consecutive pairs; gray, resized to 160 x 120; OpenCV Farneback flow):
  flow_grid_mean  16  flow magnitude per cell of a 4 x 4 grid, mean over the pairs
  flow_grid_max   16  the same, max over the pairs
  flow_global      3  mean, std, 95th percentile of the magnitude (mean over the pairs)
  flow_dir         8  histogram of flow direction weighted by magnitude, normalized
  flow_temporal    1  std over the pairs of the mean magnitude
  diff_grid       16  mean absolute gray difference per grid cell, mean over the pairs
  diff_global      1  mean absolute gray difference
  (61 values; the order is FEATURE_NAMES)
Scores (all causal: the current window, plus the previous window for the d_ inputs):
  flow_mean, diff_mean   no training: the global flow magnitude / gray difference
  motion                 BCE head on the 61 motion values (log1p, standardized)
  d_motion               BCE head on [motion_t, motion_t - motion_(t-1)]
  x                      BCE head on the encoder feature x_t (the reference)
  x_motion               BCE head on [x_t, motion_t]
Heads: window_eval.fit_bce (Holmes labels, lam on validation). x and x_motion use the encoder state with the best
validation in-video AUC.
Usage: uv run python <this file> extract
       uv run python <this file> fit <encoder features run dir> <out name> [kind]
extract writes $S1_FEATURES/encoder_probe_ucf_short_v1_f8_motion/run-1/<set>/{motion.npy, meta.jsonl}.
fit writes run-1/analysis-motion/<out name>.json and prints one line per score.
"""

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch

from pairwise_fit import in_video_auc, read_meta
from s1stream.encoder.extract import load_encoder_config
from s1stream.probe.extract import SETS, plan_sets
from s1stream.probe.fit import split
from s1stream.registry import DATASETS
from window_eval import RUN, evaluate, fit_bce, line, previous_rows, test_items

MOTION_DIR = Path(os.environ["S1_FEATURES"]) / "encoder_probe_ucf_short_v1_f8_motion" / "run-1"
SIZE, GRID, DIR_BINS = (160, 120), 4, 8
FEATURE_NAMES = ([f"flow_grid_mean_{i}" for i in range(GRID * GRID)] + [f"flow_grid_max_{i}" for i in range(GRID * GRID)]
                 + ["flow_mean", "flow_std", "flow_p95"] + [f"flow_dir_{i}" for i in range(DIR_BINS)]
                 + ["flow_temporal"] + [f"diff_grid_{i}" for i in range(GRID * GRID)] + ["diff_mean"])
WORKERS = 8


# ---------- extraction ----------

def grid_mean(image: np.ndarray) -> np.ndarray:
    """Mean of a (120, 160) array per cell of the GRID x GRID grid, row-major."""
    h, w = image.shape
    return image.reshape(GRID, h // GRID, GRID, w // GRID).mean(axis=(1, 3)).ravel()


def window_motion(frames: list[np.ndarray]) -> np.ndarray:
    """The 61 motion values of one window (see FEATURE_NAMES) from its gray frames."""
    mags, cells, hist, diffs = [], [], np.zeros(DIR_BINS), []
    for a, b in zip(frames, frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(mag)
        cells.append(grid_mean(mag))
        hist += np.histogram(ang, bins=DIR_BINS, range=(0, 2 * np.pi), weights=mag)[0]
        diffs.append(np.abs(b.astype(np.float32) - a.astype(np.float32)))
    cells, mag_mean, diff = np.array(cells), np.mean(mags, axis=0), np.mean(diffs, axis=0)
    pair_means = np.array([m.mean() for m in mags])
    return np.concatenate([
        cells.mean(0), cells.max(0),
        [mag_mean.mean(), mag_mean.std(), np.percentile(mag_mean, 95)],
        hist / (hist.sum() + 1e-8), [pair_means.std()],
        grid_mean(diff), [diff.mean()],
    ]).astype(np.float32)


def video_motion(job: tuple[str, list[tuple[int, tuple[int, ...]]]]) -> list[tuple[int, np.ndarray]]:
    """Read one video once, in order; return (row index, motion values) for each of its windows."""
    path, windows = job
    wanted = sorted({index for _, indices in windows for index in indices})
    frames, cap, position = {}, cv2.VideoCapture(path), 0
    try:
        for index in wanted:
            while position < index:
                cap.grab()
                position += 1
            ok, frame = cap.read()
            position += 1
            if not ok:
                raise RuntimeError(f"Could not read {path} at frame {index}")
            frames[index] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), SIZE, interpolation=cv2.INTER_AREA)
    finally:
        cap.release()
    return [(row, window_motion([frames[i] for i in indices])) for row, indices in windows]


def extract() -> None:
    config = load_encoder_config(RUN / "config.yaml")
    sets = plan_sets(DATASETS.build(config.train).items(), DATASETS.build(config.test).items(), config.window)
    reference = Path(os.environ["S1_FEATURES"]) / "encoder_probe_ucf_short_v1_f8" / "vjepa2-vitl" / "run-1"
    for name in SETS:
        target = MOTION_DIR / name
        if (target / "motion.npy").exists():
            print(f"{name}: exists", flush=True)
            continue
        rows = sets[name]
        meta = [json.loads(line) for line in (reference / name / "meta.jsonl").read_text().splitlines()]
        if [(m["video"], m["start_s"]) for m in meta] != [(r.item.name, r.window.start_s) for r in rows]:
            raise ValueError(f"{name}: planned windows differ from the stored feature meta")
        jobs: dict[str, list] = {}
        for i, row in enumerate(rows):
            jobs.setdefault(str(row.item.path), []).append((i, row.window.frame_indices))
        out = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
        with Pool(WORKERS) as pool:
            for done, result in enumerate(pool.imap_unordered(video_motion, jobs.items()), 1):
                for i, values in result:
                    out[i] = values
                if done % 50 == 0:
                    print(f"{name}: {done}/{len(jobs)} videos", flush=True)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "motion.npy", out)
        (target / "meta.jsonl").write_text((reference / name / "meta.jsonl").read_text())
        print(f"{name}: {len(rows)} windows written", flush=True)
    (MOTION_DIR / "features.json").write_text(json.dumps({"names": FEATURE_NAMES, "size": SIZE, "grid": GRID}) + "\n")


# ---------- fit ----------

def load_motion(set_names: tuple[str, ...]) -> np.ndarray:
    return np.log1p(np.concatenate([np.load(MOTION_DIR / s / "motion.npy") for s in set_names]))


def main_fit(features: str, name: str, kind: str = "patch_mean", device: str = "cuda") -> None:
    features_dir = Path(features)
    out = RUN / "analysis-motion" / f"{name}.json"
    if out.exists():
        raise FileExistsError(out)
    torch.cuda.set_per_process_memory_fraction(0.2)
    meta = read_meta(features_dir)
    train_meta = meta["train_anomaly"] + meta["train_normal"]
    videos = [row["video"] for row in train_meta]
    val = split(videos, {row["video"]: row["category"] for row in train_meta})
    is_val = np.array([video in val for video in videos])
    labels = np.array([row["holmes"] for row in train_meta], dtype=np.float32)
    anomaly = np.array([row["category"] != "Normal" for row in train_meta])

    train_m, test_m = load_motion(("train_anomaly", "train_normal")), load_motion(("test",))
    train_prev, test_prev = previous_rows(train_meta), previous_rows(meta["test"])
    train_dm = train_m - train_m[np.maximum(train_prev, 0)]
    train_dm[train_prev < 0] = 0
    test_dm = test_m - test_m[np.maximum(test_prev, 0)]
    test_dm[test_prev < 0] = 0

    chosen = {}
    for score in ("flow_mean", "diff_mean"):
        column = FEATURE_NAMES.index(score)
        chosen[score] = (in_video_auc(train_m[is_val, column], labels[is_val], anomaly[is_val]), None, None,
                         test_m[:, column])
    value, lam, scores = fit_bce(train_m, test_m, labels, anomaly, is_val, device)
    chosen["motion"] = (value, None, lam, scores)
    value, lam, scores = fit_bce(np.concatenate([train_m, train_dm], 1), np.concatenate([test_m, test_dm], 1),
                                 labels, anomaly, is_val, device)
    chosen["d_motion"] = (value, None, lam, scores)

    arrays = {s: np.load(features_dir / s / f"{kind}.npy", mmap_mode="r")
              for s in ("train_anomaly", "train_normal", "test")}
    best = {"x": (-np.inf, None, None, None), "x_motion": (-np.inf, None, None, None)}
    for state in range(arrays["test"].shape[1]):
        train = np.concatenate([np.asarray(arrays[s][:, state], dtype=np.float32)
                                for s in ("train_anomaly", "train_normal")])
        test = np.asarray(arrays["test"][:, state], dtype=np.float32)
        inputs = {"x": (train, test),
                  "x_motion": (np.concatenate([train, train_m], 1), np.concatenate([test, test_m], 1))}
        for input_name, (tr, te) in inputs.items():
            value, lam, scores = fit_bce(tr, te, labels, anomaly, is_val, device)
            best[input_name] = max(best[input_name], (value, state, lam, scores), key=lambda b: b[0])
        print(f"state {state} done", flush=True)
    chosen.update(best)

    items = test_items()
    results = {}
    for score, (value, state, lam, values) in chosen.items():
        results[score] = {"state": state, "lam": lam, "validation_in_video": value,
                          **evaluate(values, meta["test"], items)}
        print(line(f"{name}/{score} state {state} val {value:.3f}", results[score]), flush=True)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"features": str(features_dir), "kind": kind, "motion": str(MOTION_DIR),
                               "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract()
    else:
        main_fit(*sys.argv[2:])
