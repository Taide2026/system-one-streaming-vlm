#!/usr/bin/env python3
"""Other small vision-language models, zero-shot, at 1x and 2x (2026-11-14): does the gain of enlarging depend on whether
the vision tower gives MORE TOKENS for a larger input?

Any Hugging Face image-text-to-text model: one user turn with the window's frames and the same prompt the plain Qwen
read-out uses (state, question, options, "Answer with one option only."), the model's own chat template and its
processor at their DEFAULT settings; the score is softmax over the first tokens of "Yes" and "No" at the answer
position. Writes scores.jsonl like the other runs, plus per window the visual tokens the model actually received
(sequence length minus the length of the same prompt without images), which is the quantity of interest.

  python scripts/smallvlm.py --model <dir> --config <eval yaml> --out <run dir> --factor 1|2 [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import s1stream.plugins  # noqa: F401  registers the datasets
from s1stream.backbones.qwen_first_token import build_prompt
from s1stream.config import load_config
from s1stream.input_forms.frames import read_frames
from s1stream.registry import DATASETS
from s1stream.windows import plan_windows

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True); ap.add_argument("--config", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--factor", type=float, default=1.0); ap.add_argument("--via", type=int, default=0); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--probe", type=int, default=0)
args = ap.parse_args()
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

config = load_config(args.config); q = config.question
proc = AutoProcessor.from_pretrained(args.model, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
tok = proc.tokenizer
# the answer may start with or without a leading space, depending on the chat template: both variants are prepared and the
# one that carries the probability mass on the first windows is used (printed)
CAND = {"plain": [tok.encode(o, add_special_tokens=False)[0] for o in q.options], "space": [tok.encode(" " + o, add_special_tokens=False)[0] for o in q.options]}
CAND = {k: v for k, v in CAND.items() if len(set(v)) == len(v)}
ids = CAND["plain"]; PICK = {"n": 0, "mass": {k: 0.0 for k in CAND}}


def render(n_images: int) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}] * n_images + [{"type": "text", "text": build_prompt(q)}]}]
    try:
        return proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


TEXT_LEN = len(tok(render(0), add_special_tokens=True)["input_ids"])


def encode(images: list) -> dict:
    text = render(len(images))
    for imgs in (list(images), [list(images)]):
        try:
            return proc(text=[text], images=imgs, return_tensors="pt")
        except Exception as exc:  # some processors want one list of images per prompt
            err = exc
    raise err


def run(images: list) -> tuple[float, int, float]:
    if args.via:        # as input_forms/soften.py: same size and tokens, less detail (2026-11-16)
        from s1stream.input_forms.soften import soften
        images = soften(images, args.via)
    if args.factor != 1:
        images = [Image.fromarray(cv2.resize(np.asarray(im.convert('RGB')), (int(im.size[0] * args.factor), int(im.size[1] * args.factor)), interpolation=cv2.INTER_CUBIC)) for im in images]   # as input_forms/upsample.py
    enc = encode(images)
    enc = {k: (v.to("cuda", torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v.to("cuda") if torch.is_tensor(v) else v) for k, v in enc.items()}
    with torch.inference_mode():
        logits = model(**enc).logits[0, -1].float()
    global ids
    if PICK["n"] < 8:
        p = torch.softmax(logits, -1)
        for k, v in CAND.items():
            PICK["mass"][k] += float(p[v].sum())
        PICK["n"] += 1; best = max(PICK["mass"], key=PICK["mass"].get)
        if PICK["n"] == 8:
            ids = CAND[best]; print("answer tokens:", best, ids, [tok.decode([i]) for i in ids], {k: round(v / 8, 3) for k, v in PICK["mass"].items()}, flush=True)
    mass = float(torch.softmax(logits, -1)[ids].sum())
    return float(logits[ids[0]] - logits[ids[1]]), int(enc["input_ids"].shape[1]) - TEXT_LEN, mass


items = list(DATASETS.build(config.dataset).items())
if args.probe:      # token counts and answer mass on a few windows, both factors, then stop
    for f in (1.0, 2.0):
        args.factor = f
        for item in items[:args.probe]:
            w = plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames)[2]
            ims = read_frames(item.path, w.frame_indices); m, n, mass = run(ims); mass = max(v / PICK["n"] for v in PICK["mass"].values())
            print(f"PROBE {args.model.name} x{f:g} frame {ims[0].size} visual tokens {n} ({n / len(ims):.0f}/frame) margin {m:+.2f} p(Yes)+p(No) {mass:.3f}", flush=True)
    sys.exit(0)
for item in items[:8]:                                  # decide the answer-token variant before anything is scored
    run(read_frames(item.path, plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames)[0].frame_indices))
args.out.mkdir(parents=True, exist_ok=True)             # resumable: windows already in scores.jsonl are skipped
done = set()
if (args.out / "scores.jsonl").is_file():
    rows = [json.loads(l) for l in (args.out / "scores.jsonl").read_text().splitlines() if l.strip().endswith("}")]
    done = {(r["video"], r["window"]) for r in rows}
    (args.out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows)); print("resuming after", len(done), "windows", flush=True)
(args.out / "config.yaml").write_text(json.dumps({"model": str(args.model), "factor": args.factor, "via": args.via, "config": str(args.config), "text_len": TEXT_LEN, "answer_ids": ids}))
n = len(done)
with (args.out / "scores.jsonl").open("a") as stream:
    for item in items:
        for w in plan_windows(item, config.window.length_s, config.window.stride_s, config.window.frames):
            if (item.name, w.index) in done:
                continue
            m, nt, mass = run(read_frames(item.path, w.frame_indices))
            stream.write(json.dumps({"video": item.name, "window": w.index, "start_s": w.start_s, "end_s": w.end_s, "label": w.label,
                                     "score": 1 / (1 + math.exp(-m)), "margin": m, "tokens": nt, "mass": round(mass, 4)}) + "\n")
            n += 1
            if n % 200 == 0:
                stream.flush()
            if n % 2000 == 0:
                pass; print(f"{n} windows", flush=True)
            if args.limit and n >= args.limit:
                break
        if args.limit and n >= args.limit:
            break
(args.out / "run.json").write_text(json.dumps({"windows": n, "finished": datetime.now(timezone.utc).isoformat()}))
print("SMALLVLM_DONE", n, flush=True)
