#!/usr/bin/env python3
"""Score windows of 8 full frames with a frozen d3-nano, with or without a 2x bicubic up-scale (2026-11-06,
full-frame-sr proposal, decisions 1 and 5).

  uv run --frozen python scripts/sr-score.py --frames-dir <frames dir> --index <windows.jsonl> --upscale bicubic2 \\
      --out <dir>/B/scores.jsonl [--limit-windows N]

Conditions: B = original frames + --upscale bicubic2; C = SR frames (scripts/sr-frames.py) + --upscale none.
Original frames + --upscale none is the draft check: it must reproduce the saved zero-shot scores.
One scores.jsonl line per window: video, window, start_s, end_s, label, score = p(Yes), hot_area, seconds, size.
Resumable: skips windows already in --out. Writes config.yaml next to --out.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import s1stream.plugins  # noqa: E402,F401
from s1stream.assets import resolve  # noqa: E402
from s1stream.backbones.d3 import QUESTION_ID, build_questions  # noqa: E402
from s1stream.prompt import PROMPTS  # noqa: E402
from s1stream.registry import BACKBONES  # noqa: E402
from s1stream.types import ModelInput, Question  # noqa: E402


def keep(frame: np.ndarray) -> np.ndarray:
    return frame


def bicubic2(frame: np.ndarray) -> np.ndarray:
    """2x per side with cubic interpolation (the same call as the classmate's crop up-scale)."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (2 * w, 2 * h), interpolation=cv2.INTER_CUBIC)


# Scale and interpolation sweep (Eric, 2026-11-10): "<method><factor>", e.g. bicubic1.5, lanczos2.
# The new size is round(factor * side). bicubic2 keeps its own function, so earlier runs stay comparable.
INTERPOLATION = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR,
                 "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}
FACTORS = ("1.25", "1.5", "1.75", "2", "3")


def resizer(method: str, factor: str):
    """A function that resizes one frame by factor with the given cv2 interpolation."""
    def resize(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        size = (round(float(factor) * w), round(float(factor) * h))
        return cv2.resize(frame, size, interpolation=INTERPOLATION[method])
    return resize


UPSCALE = {"none": keep}
for _method in INTERPOLATION:
    for _factor in FACTORS:
        UPSCALE[f"{_method}{_factor}"] = resizer(_method, _factor)
UPSCALE["bicubic2"] = bicubic2
# odd-bicubic2: frames 1, 3, 5, 7 at bicubic 2x, frames 0, 2, 4, 6 unchanged. This is the training input of
# fovfull2x_d3nano_c (foveate with frac 1.0 and upscale 2: the "crop" is the whole frame). Added 2026-11-09.
UPSCALE_CHOICES = sorted(UPSCALE) + ["odd-bicubic2"]


def window_images(frames: np.ndarray, upscale: str) -> list[Image.Image]:
    """The model input of one window: frames [8, H, W, 3]."""
    images = []
    for index, frame in enumerate(frames):
        frame = np.ascontiguousarray(frame)
        if upscale == "odd-bicubic2":
            frame = bicubic2(frame) if index % 2 == 1 else frame
        else:
            frame = UPSCALE[upscale](frame)
        images.append(Image.fromarray(frame))
    return images


def image_score(backbone, images: list[Image.Image], question: Question) -> tuple[float, dict]:
    """p(Yes) with the 8 frames sent as 8 separate images (the d3 backbone's images= path)."""
    logits = backbone.forward(ModelInput(images=images, text=None), question).logits
    positive = question.options.index(question.positive)
    return float(logits.detach().float().cpu().softmax(dim=-1)[positive]), {}


def video_score(backbone, images: list[Image.Image], question: Question) -> tuple[float, dict]:
    """p(Yes) with the 8 frames sent as one video (d3 videos= path). Added 2026-11-09.

    The runtime takes the frames as sampled at 2 per second, joins every 2 frames into one token group with a
    timestamp, and shrinks a frame above 200,704 pixels (0.2 MP) to that size.
    """
    result = backbone.model.system_one(state=question.state, videos=[images], questions=build_questions(question))
    answer = result["answers"][QUESTION_ID]
    if "error" in answer:
        raise RuntimeError(f"d3 could not answer: {answer['error']}: {answer.get('message', '')}")
    probabilities = answer["probabilities"]
    total = sum(probabilities[option] for option in question.options)
    return probabilities[question.positive] / total, result.get("usage", {})


SCORE = {"images": image_score, "video": video_score}


def read_done(path: Path) -> set[tuple[str, int]]:
    """(video, window) pairs already scored in path."""
    if not path.exists():
        return set()
    return {(r["video"], r["window"]) for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--upscale", choices=UPSCALE_CHOICES, required=True)
    ap.add_argument("--send-as", choices=sorted(SCORE), default="images",
                    help="images: 8 separate images (all runs before 2026-11-09); video: one 8-frame video")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-windows", type=int, default=None)
    ap.add_argument("--asset", default="d3-nano")
    ap.add_argument("--memory-cap-gb", type=float, default=40)
    ap.add_argument("--adapter", default=None, help="fine-tuned run dir (adapter/, readout.safetensors, calibration.json)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.index.read_text().splitlines()]
    if args.limit_windows is not None:
        rows = rows[:args.limit_windows]
    done = read_done(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) for k, v in vars(args).items()} | {"torch": str(torch.__version__),
                                                           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    (args.out.parent / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    p = PROMPTS["v1"]
    question = Question(state=p["state"], text=p["text"], options=tuple(p["options"]), positive=p["positive"])
    t_load = time.perf_counter()
    spec = {"name": "d3", "asset": args.asset, "path": str(resolve(args.asset)), "memory_cap_gb": args.memory_cap_gb}
    if args.adapter:
        spec["adapter"] = args.adapter
    backbone = BACKBONES.build(spec)
    print(f"loaded {args.asset} in {time.perf_counter() - t_load:.1f} s; upscale {args.upscale}; "
          f"{len(rows) - len(done)} windows to score", flush=True)

    video, frames = None, None
    n, t0 = 0, time.perf_counter()
    with args.out.open("a") as out:
        for r in rows:
            if (r["video"], r["window"]) in done:
                continue
            if r["video"] != video:
                video = r["video"]
                frames = np.load(args.frames_dir / f"{video}.npy", mmap_mode="r")
            images = window_images(frames[r["window"]], args.upscale)
            t = time.perf_counter()
            with torch.inference_mode():
                score, usage = SCORE[args.send_as](backbone, images, question)
            seconds = time.perf_counter() - t
            out.write(json.dumps({"video": r["video"], "window": r["window"], "start_s": r["start_s"],
                                  "end_s": r["end_s"], "label": r["label"], "score": score,
                                  "hot_area": r["hot_area"], "seconds": seconds,
                                  "size": list(images[0].size), "usage": usage}) + "\n")
            n += 1
            if n % 500 == 0:
                out.flush()
                print(f"{n} windows, {(time.perf_counter() - t0) / n:.3f} s/window", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"done: {n} windows in {time.perf_counter() - t0:.0f} s; peak GPU {peak:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
