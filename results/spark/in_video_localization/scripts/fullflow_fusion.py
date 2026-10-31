"""Held-out check of the score + flow fusion (fusion_anchor.py) on the full 290-video UCF-Crime test set.

Question: on the 117 short test videos, z(head) + flow with w = 0.2-0.3 raised the in-video AUC without lowering the
overall AUC, but w was picked on those test videos. Here w is chosen on the 117 short test videos and checked once
on the other 173 test videos, which no probe experiment used. Scores: the zero-shot and fine-tuned VLM runs of
finetune_in_video_auc.py (all 290 videos; the encoder heads exist only for the 117 short videos).
Steps:
  extract   pixel motion (motion_fit.window_motion) of every window of the 290 videos, frames as plan_windows
  fuse      per run: score = (1 - w) * z(logit p(yes)) + w * z(log1p flow_mean); z() uses the 117-video windows
            selection rule (fixed before the held-out run): the w in WEIGHTS with the highest in-video AUC among the
            w whose overall AUC is not below that of w = 0, on the 117 short videos
            report on the 173 held-out videos: w = 0, the selected w, and w = 0.2 (window_eval.evaluate)
Usage: uv run python <this file> extract
       uv run python <this file> fuse
extract writes $S1_FEATURES/ucf_test_v1_f8_motion/run-1/{motion.npy, meta.jsonl};
fuse writes in_video_localization/07-fusion/full-test-heldout.json and prints one line per run and w.
"""

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from paths import FUSION
import finetune_in_video_auc
from motion_fit import FEATURE_NAMES, WORKERS, video_motion
from s1stream.windows import plan_windows
from window_eval import evaluate, line

OUT_DIR = Path(os.environ["S1_FEATURES"]) / "ucf_test_v1_f8_motion" / "run-1"
SHORT_META = Path(os.environ["S1_FEATURES"]) / "encoder_probe_ucf_short_v1_f8" / "vjepa2-vitl" / "run-1" / "test" / "meta.jsonl"
WEIGHTS = tuple(round(0.1 * i, 1) for i in range(11))


def key(row: dict) -> tuple[str, float]:
    return row["video"], round(row["start_s"], 3)


def extract() -> None:
    if (OUT_DIR / "motion.npy").exists():
        print("exists")
        return
    items = finetune_in_video_auc.test_items()
    _, path = next(finetune_in_video_auc.runs())
    rows = finetune_in_video_auc.read_jsonl(path)
    frames = {}
    for item in items.values():
        for window in plan_windows(item, 2.0, 2.0, 8):
            frames[(item.name, round(window.start_s, 3))] = window.frame_indices
    jobs: dict[str, list] = {}
    for i, row in enumerate(rows):
        jobs.setdefault(str(items[row["video"]].path), []).append((i, frames[key(row)]))
    out = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    with Pool(WORKERS) as pool:
        for done, result in enumerate(pool.imap_unordered(video_motion, jobs.items()), 1):
            for i, values in result:
                out[i] = values
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} videos", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "motion.npy", out)
    (OUT_DIR / "meta.jsonl").write_text("".join(json.dumps({k: row[k] for k in ("video", "start_s", "end_s")}) + "\n"
                                                for row in rows))
    print(f"{len(rows)} windows written")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def select_weight(curve: dict[float, dict]) -> float:
    """Highest in-video AUC among the w whose overall AUC is not below that of w = 0."""
    floor = curve[0.0]["all"]
    allowed = [w for w in WEIGHTS if curve[w]["all"] >= floor]
    return max(allowed, key=lambda w: curve[w]["in_video"])


def fuse() -> None:
    out = FUSION / "full-test-heldout.json"
    if out.exists():
        raise FileExistsError(out)
    items = finetune_in_video_auc.test_items()
    motion_meta = finetune_in_video_auc.read_jsonl(OUT_DIR / "meta.jsonl")
    flow_by_key = dict(zip(map(key, motion_meta),
                           np.log1p(np.load(OUT_DIR / "motion.npy")[:, FEATURE_NAMES.index("flow_mean")]), strict=True))
    short = {row["video"] for row in finetune_in_video_auc.read_jsonl(SHORT_META)}
    print(f"short videos {len(short)}, held-out videos {len(set(items) - short)}", flush=True)

    results = {}
    for label, path in finetune_in_video_auc.runs():
        rows = finetune_in_video_auc.read_jsonl(path)
        for row in rows:
            row["category"] = items[row["video"]].category
        score = logit(np.array([row["score"] for row in rows], dtype=np.float64))
        flow = np.array([flow_by_key[key(row)] for row in rows])
        is_short = np.array([row["video"] in short for row in rows])
        z_score = (score - score[is_short].mean()) / score[is_short].std()
        z_flow = (flow - flow[is_short].mean()) / flow[is_short].std()
        parts = {"short": [r for r, s in zip(rows, is_short, strict=True) if s],
                 "heldout": [r for r, s in zip(rows, is_short, strict=True) if not s]}
        fused = {w: (1 - w) * z_score + w * z_flow for w in WEIGHTS}
        curve = {w: evaluate(fused[w][is_short], parts["short"], items) for w in WEIGHTS}
        chosen = select_weight(curve)
        heldout = {f"w{w:g}": evaluate(fused[w][~is_short], parts["heldout"], items) for w in sorted({0.0, chosen, 0.2})}
        results[label] = {"selected_w": chosen, "short_curve": {str(w): v for w, v in curve.items()}, "heldout": heldout}
        for name, value in heldout.items():
            print(line(f"{label} {name}{' (selected)' if name == f'w{chosen:g}' else ''}", value), flush=True)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    {"extract": extract, "fuse": fuse}[sys.argv[1]]()
