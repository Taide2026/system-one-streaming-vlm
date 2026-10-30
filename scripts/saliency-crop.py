#!/usr/bin/env python3
"""Self-localised crop (evidence b, 2026-10-30): the model's own gradient saliency picks the region, ViCrop-style.

  uv run --frozen python scripts/saliency-crop.py --manifest <eval manifest.json> --root <ucf-crime asset>
      [--adapter <ft run>] --out <json> [--events|--windows video:start_s,...]

One grad-enabled draft pass per window (the d3 release's own image path, as the trainee): embeddings -> encoder ->
last position -> code read-out; saliency = ||d(z_Yes - z_No) / d embeds|| per visual token, summed over the 8 frames'
aligned grid cells; the peak cell centres a 1/3-of-frame box; the crop (2x up-scaled) is scored with a normal
inference pass. Prints p_draft, p_saliency_crop and the peak cell per window; TGM comparison comes from zoom-pilot /
zoom-full on the same windows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import s1stream.plugins  # noqa: E402,F401
from s1stream.assets import resolve  # noqa: E402
from s1stream.backbones.d3 import build_questions, load_adapter, load_release_code, runtime_device  # noqa: E402
from s1stream.prompt import PROMPTS  # noqa: E402
from s1stream.types import Question  # noqa: E402

EVENTS = {"RoadAccidents010_x264.mp4": (7.7, 9.03), "RoadAccidents004_x264.mp4": (4.7, 6.33),
          "RoadAccidents011_x264.mp4": (8.7, 10.03), "RoadAccidents122_x264.mp4": (10.0, 12.03),
          "Shoplifting037_x264.mp4": (38.0, 40.03)}


def frames_at(path, times, fps):
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, fr = cap.read()
            if ok:
                out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return out


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--asset", default="d3-nano")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    man = {v["name"]: v for v in json.loads(args.manifest.read_text())}
    path = resolve(args.asset)
    code = load_release_code(path)
    runtime = code.D3.from_pretrained(path, device=runtime_device("cuda"), local_files_only=True)
    if args.adapter:
        load_adapter(runtime, Path(args.adapter))
    p = PROMPTS["v1"]
    question = Question(state=p["state"], text=p["text"], options=tuple(p["options"]), positive=p["positive"])
    rendered = code.normalize_question(build_questions(question)["answer"]).rendered
    image_id = runtime.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    device = next(runtime.backbone.parameters()).device

    def saliency_pass(images: list[Image.Image]):
        text = runtime.image_text(question.state, rendered, len(images))
        inputs = {k: v.to(device) for k, v in runtime.processor(text=[text], images=images, return_tensors="pt").items()}
        with code.sdpa_backends(device):
            embeds, mrope = runtime.image_embeddings(inputs)
            embeds = embeds.detach().requires_grad_(True)
            hidden = (runtime.encoder(embeds, inputs["attention_mask"], mrope)[:, -1] if runtime.encoder is not None
                      else runtime.backbone(inputs_embeds=embeds, attention_mask=inputs["attention_mask"], use_cache=False).last_hidden_state[:, -1])
            logits = hidden.float() @ runtime.readout.float().T
            (logits[0, 0] - logits[0, 1]).backward()
        sal = embeds.grad[0].float().norm(dim=-1).cpu().numpy()
        vis = (inputs["input_ids"][0] == image_id).cpu().numpy()
        grid = inputs["image_grid_thw"].cpu().numpy()  # one row per image: (t, h, w) in patches
        p_draft = torch.softmax(logits[0, :2] / runtime.temperature, dim=-1)[0].item()
        return sal[vis], grid, p_draft

    def plain_score(images):
        text = runtime.image_text(question.state, rendered, len(images))
        inputs = {k: v.to(device) for k, v in runtime.processor(text=[text], images=images, return_tensors="pt").items()}
        with torch.inference_mode(), code.sdpa_backends(device):
            embeds, mrope = runtime.image_embeddings(inputs)
            hidden = (runtime.encoder(embeds, inputs["attention_mask"], mrope)[:, -1] if runtime.encoder is not None
                      else runtime.backbone(inputs_embeds=embeds, attention_mask=inputs["attention_mask"], use_cache=False).last_hidden_state[:, -1])
            logits = hidden.float() @ runtime.readout.float().T
        return torch.softmax(logits[0, :2] / runtime.temperature, dim=-1)[0].item()

    results = {}
    for name, (t0, t1) in EVENTS.items():
        m = man[name]; fps = m["fps"]; vpath = args.root / m["path"]
        frames = frames_at(vpath, [t0 + (k + 0.5) * (t1 - t0) / 8 for k in range(8)], fps)
        images = [Image.fromarray(f) for f in frames]
        sal, grid, p_draft = saliency_pass(images)
        # per-image token grids (merge 2x2): h/2 x w/2 cells, raster order; sum the 8 frames' aligned grids
        heat, off = None, 0
        for (t, h, w) in grid:
            hh, ww = int(h) // 2, int(w) // 2
            cells = sal[off:off + hh * ww].reshape(hh, ww)
            heat = cells if heat is None else heat + cells
            off += hh * ww
        r, c = np.unravel_index(int(heat.argmax()), heat.shape)
        fy, fx = (r + 0.5) / heat.shape[0], (c + 0.5) / heat.shape[1]
        H, W = frames[0].shape[:2]
        h3, w3 = H // 3, W // 3
        y0, x0 = min(max(int(fy * H) - h3 // 2, 0), H - h3), min(max(int(fx * W) - w3 // 2, 0), W - w3)
        crop = [Image.fromarray(cv2.resize(f[y0:y0 + h3, x0:x0 + w3], (w3 * 2, h3 * 2), interpolation=cv2.INTER_CUBIC)) for f in frames]
        p_crop = plain_score(crop)
        results[name] = {"p_draft": p_draft, "p_saliency_crop": p_crop, "peak": [float(fy), float(fx)]}
        print(name[:-9], {k: (round(v, 3) if isinstance(v, float) else [round(x, 2) for x in v]) for k, v in results[name].items()}, flush=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
