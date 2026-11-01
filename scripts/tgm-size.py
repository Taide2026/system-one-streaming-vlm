#!/usr/bin/env python3
"""Event spatial-size proxy from TGM (2026-11-01), for the jev-hard subset: how much of the frame moves during each
annotated UCF test event. No model - pure OpenCV, CPU.

  python scripts/tgm-size.py --manifest <manifest.json> --root <ucf-crime asset> --out <jsonl>

Per event: up to 5 probe windows spread across the event span; per window the TGM heat map of 8 frames (the recipe
of scripts/zoom-full.py: grey, 5x5 blur, summed |frame diffs|, 31x31 blur); the hot-area fraction = share of pixels
above half the peak. The event's area_frac = the median over its probe windows - small area_frac = a small / local
event (the case-3 regime), large = scene-wide motion. Output: one line per event {video, event, area_frac, tgm_peak},
then terciles printed. The subset definition is independent of any model or policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def heat_of(frames: list[np.ndarray]) -> np.ndarray:
    grey = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32) for f in frames]
    return cv2.GaussianBlur(np.sum([np.abs(a - b) for a, b in zip(grey[1:], grey[:-1])], axis=0), (31, 31), 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probes", type=int, default=5)
    args = ap.parse_args()
    man = json.loads(args.manifest.read_text())
    rows = []
    for m in man:
        if not m.get("events"):
            continue
        cap = cv2.VideoCapture(str(args.root / m["path"]))
        fps = m["fps"]
        for e0, e1 in m["events"]:
            t0, t1 = e0 / fps, (e1 + 1) / fps
            n = max(1, min(args.probes, int((t1 - t0) / 2)))
            fracs, peaks = [], []
            for j in range(n):
                w0 = t0 + (j + 0.5) * (t1 - t0) / n - 1.0  # a 2 s probe window centred in slice j
                frames = []
                for k in range(8):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(max(w0 + (k + 0.5) * 0.25, 0) * fps)))
                    ok, fr = cap.read()
                    if ok:
                        frames.append(fr)
                if len(frames) < 4:
                    continue
                heat = heat_of(frames)
                peak = float(heat.max())
                if peak <= 0:
                    continue
                fracs.append(float((heat > 0.5 * peak).mean()))
                peaks.append(peak)
            if fracs:
                rows.append({"video": m["name"], "event": [e0, e1],
                             "area_frac": float(np.median(fracs)), "tgm_peak": float(np.median(peaks))})
        cap.release()
        print(f"{m['name']}: {len(m['events'])} events", flush=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    a = np.array([r["area_frac"] for r in rows])
    t1, t2 = np.quantile(a, [1 / 3, 2 / 3])
    print(f"{len(rows)} events; area_frac terciles: small < {t1:.4f} <= medium < {t2:.4f} <= large")
    for name, mask in [("small", a < t1), ("medium", (a >= t1) & (a < t2)), ("large", a >= t2)]:
        print(f"  {name}: {int(mask.sum())} events, mean area {a[mask].mean():.4f}")


if __name__ == "__main__":
    main()
