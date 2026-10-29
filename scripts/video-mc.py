#!/usr/bin/env python3
"""Multiple-choice video benchmarks in one pass (2026-10-29): MVBench and Video-MME on a decision backbone.

  uv run --frozen python scripts/video-mc.py --bench mvbench --root <mvbench dir> --backbone d3 --asset d3-nano
      [--adapter <run dir>] --out <results dir> [--tasks a,b] [--limit N] [--frames 8]

Per question: `frames` frames spread evenly over the clip (or its [start, end] when the task gives bounds, or over a
folder of frames), the question text and the candidates as the typed options; the answer is the option with the
highest probability (backbone.forward, no decoding). Writes <out>/answers.jsonl (one line per question: task, id,
probabilities, answer, correct) and <out>/summary.json (accuracy per task and overall). MVBench: json/<task>.json +
video/ unpacked as OpenGVLab/MVBench ships them (the task -> folder table below is the benchmark's own); Video-MME:
videomme/test-*.parquet + videos/ unpacked, `--subset short` picks the short videos.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import s1stream.plugins  # noqa: E402,F401
from s1stream.assets import resolve  # noqa: E402
from s1stream.registry import BACKBONES  # noqa: E402
from s1stream.types import ModelInput, Question  # noqa: E402

# MVBench: task -> (json, video folder, kind, has bounds); from the benchmark's evaluation notebook.
MVBENCH = {
    "action_sequence": ("star/Charades_v1_480", "video", True), "action_prediction": ("star/Charades_v1_480", "video", True),
    "action_antonym": ("ssv2_video", "video", False), "fine_grained_action": ("Moments_in_Time_Raw/videos", "video", False),
    "unexpected_action": ("FunQA_test/test", "video", False), "object_existence": ("clevrer/video_validation", "video", False),
    "object_interaction": ("star/Charades_v1_480", "video", True), "object_shuffle": ("perception/videos", "video", False),
    "moving_direction": ("clevrer/video_validation", "video", False), "action_localization": ("sta/sta_video", "video", True),
    "scene_transition": ("scene_qa/video", "video", False), "action_count": ("perception/videos", "video", False),
    "moving_count": ("clevrer/video_validation", "video", False), "moving_attribute": ("clevrer/video_validation", "video", False),
    "state_change": ("perception/videos", "video", False), "fine_grained_pose": ("nturgbd", "video", False),
    "character_order": ("perception/videos", "video", False), "egocentric_navigation": ("vlnqa", "video", False),
    "episodic_reasoning": ("tvqa/frames_fps3_hq", "frame", True), "counterfactual_inference": ("clevrer/video_validation", "video", False),
}
STATE = "A short video clip."


def video_frames(path: Path, n: int, start: float | None, end: float | None) -> list:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        a = int(round((start or 0.0) * fps)); b = int(round(end * fps)) - 1 if end else total - 1
        a, b = max(a, 0), max(min(b, total - 1), a)
        images = []
        for k in range(n):
            idx = int(round(a + (k + 0.5) * (b - a + 1) / n))
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, b))
            ok, frame = cap.read()
            if not ok:
                continue
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return images
    finally:
        cap.release()


def folder_frames(path: Path, n: int, start: float | None, end: float | None, fps: float = 3.0) -> list:
    files = sorted(path.glob("*.jpg")) + sorted(path.glob("*.png"))
    if start is not None and end is not None:
        files = files[int(start * fps):max(int(end * fps), int(start * fps) + 1)] or files
    idx = [int(round((k + 0.5) * len(files) / n)) for k in range(n)]
    return [Image.open(files[min(i, len(files) - 1)]).convert("RGB") for i in idx]


def mvbench_items(root: Path, tasks: list[str] | None):
    for task, (folder, kind, bounded) in MVBENCH.items():
        if tasks and task not in tasks:
            continue
        for i, r in enumerate(json.loads((root / "json" / f"{task}.json").read_text())):
            path = root / "video" / folder / r["video"]
            yield {"task": task, "id": f"{task}/{i}", "path": path, "kind": kind, "question": r["question"],
                   "options": list(r["candidates"]), "answer": r["answer"],
                   "start": r.get("start") if bounded else None, "end": r.get("end") if bounded else None}


def videomme_items(root: Path, subset: str | None):
    import pyarrow.parquet as pq

    t = pq.read_table(str(next((root / "videomme").glob("test-*.parquet")))).to_pylist()
    for r in t:
        if subset and r["duration"] != subset:
            continue
        raw = r["options"]
        if isinstance(raw, str):  # the hub parquet stores the list as its repr
            import ast
            raw = ast.literal_eval(raw)
        opts = [o.split(". ", 1)[1] if ". " in o[:4] else o for o in raw]  # "A. text" -> text
        ans = r["answer"].strip()
        answer = opts["ABCD".index(ans)] if ans in "ABCD" and len(ans) == 1 else ans
        yield {"task": r["duration"], "id": r["question_id"], "path": root / "videos" / f"{r['videoID']}.mp4", "kind": "video",
               "question": r["question"], "options": opts, "answer": answer, "start": None, "end": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["mvbench", "videomme"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--backbone", default="d3"); ap.add_argument("--asset", default="d3-nano"); ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--tasks", default=None); ap.add_argument("--subset", default=None)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()
    spec = {"name": args.backbone, "asset": args.asset, "path": str(resolve(args.asset)), "memory_cap_gb": 40}
    if args.adapter:
        spec["adapter"] = args.adapter
    backbone = BACKBONES.build(spec)
    items = mvbench_items(args.root, args.tasks.split(",") if args.tasks else None) if args.bench == "mvbench" \
        else videomme_items(args.root, args.subset)
    args.out.mkdir(parents=True, exist_ok=True)
    per: dict[str, list[int]] = {}
    n, t0, skipped = 0, time.perf_counter(), 0
    with (args.out / "answers.jsonl").open("w") as f:
        for it in items:
            if args.limit and n >= args.limit:
                break
            if not it["path"].exists():
                skipped += 1
                continue
            images = folder_frames(it["path"], args.frames, it["start"], it["end"]) if it["kind"] == "frame" \
                else video_frames(it["path"], args.frames, it["start"], it["end"])
            if not images:
                skipped += 1
                continue
            q = Question(state=STATE, text=it["question"], options=tuple(it["options"]), positive=it["options"][0])
            import torch
            with torch.inference_mode():
                out = backbone.forward(ModelInput(images=images, text=None), q)
            logits = np.asarray(out.logits, dtype=np.float32).flatten()[:len(it["options"])]
            p = np.exp(logits - logits.max()); p /= p.sum()
            pred = it["options"][int(p.argmax())]
            ok = int(pred.strip().lower() == it["answer"].strip().lower())
            per.setdefault(it["task"], []).append(ok)
            f.write(json.dumps({"task": it["task"], "id": it["id"], "probabilities": dict(zip(it["options"], map(float, p))),
                                "answer": pred, "truth": it["answer"], "correct": ok}) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"{n} questions, {(time.perf_counter() - t0) / n:.2f} s each, acc so far {100 * np.mean([x for v in per.values() for x in v]):.1f} %", flush=True)
    summary = {"n": n, "skipped": skipped, "overall": 100 * float(np.mean([x for v in per.values() for x in v])) if n else None,
               "per_task": {k: {"n": len(v), "acc": 100 * float(np.mean(v))} for k, v in per.items()}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
