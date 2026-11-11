"""Causal smoothing of window scores, per video in window order (2026-11-11).

Usage: smooth.py <scores.jsonl> <out.jsonl> mean2|ema85
mean2: mean of this window and the previous one. ema85: m = 0.85 * score + 0.15 * m_previous.
Same rules as results/spark/training_free_routes_ucf290/20261110-130100/tf.py (r_smooth k=2, r_ema a=0.85).
"""
import json
import sys
from pathlib import Path


def mean2(scores: list[float]) -> list[float]:
    return [scores[0]] + [(a + b) / 2 for a, b in zip(scores, scores[1:])]


def ema85(scores: list[float]) -> list[float]:
    out = [scores[0]]
    for s in scores[1:]:
        out.append(0.85 * s + 0.15 * out[-1])
    return out


RULES = {"mean2": mean2, "ema85": ema85}

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
by_video: dict[str, list[dict]] = {}
for r in rows:
    by_video.setdefault(r["video"], []).append(r)
with Path(sys.argv[2]).open("w") as out:
    for video_rows in by_video.values():
        video_rows.sort(key=lambda r: r["window"])
        for r, s in zip(video_rows, RULES[sys.argv[3]]([r["score"] for r in video_rows])):
            out.write(json.dumps(r | {"score": s}) + "\n")
