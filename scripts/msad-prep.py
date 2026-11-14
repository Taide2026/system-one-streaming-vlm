#!/usr/bin/env python3
"""MSAD test videos -> the frames our windows read, at several real resolutions (2026-11-13). CPU only.

Same scheme as cvad-prep.py: for every 2 s window the 8 sampled frames (time (k + 0.5) / 4 s) are decoded once from the
native file and written, area-downscaled, to one Motion-JPEG file per resolution (8 fps, each sampled frame twice;
stored frame j is native time j / 8 s). MSAD sources are mixed (mostly 1280x720, some 1920x1080, a few other widths),
so a variant fixes the HEIGHT (180 / 360 / 720) and keeps the aspect ratio, the width rounded to a multiple of 16.
A source lower than a variant's height is not enlarged for it: that video is skipped for that variant and listed.

  python3 msad-prep.py <root> [--jobs 12]     reads the Protocol-ii test list, writes <root>/prep/v{180,360,720}/<name>.avi
                                              and <root>/prep/meta/<name>.json
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

HEIGHTS = {"v180": 180, "v360": 360, "v720": 720}


def test_videos(root: Path) -> list:
    out = []
    for line in (root / "MSAD_I3D_WS_Test.list").read_text().split():
        name = Path(line).name.replace("_i3d.npy", "")
        if name.startswith("MSAD_normal_testing"):
            out.append((name, root / "MSAD_normal_testing_videos" / f"{name}.mp4"))
        else:
            out.append((name, root / "MSAD_anomaly" / name.rsplit("_", 1)[0] / f"{name}.mp4"))
    return out


def one(name: str, path: str, root: str) -> str:
    prep = Path(root) / "prep"; meta = prep / "meta" / f"{name}.json"
    if meta.is_file():
        return f"have {name}"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return f"MISSING {name} {path}"
    fps, n = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    windows = int((n / fps + 1e-9) // 2)
    if windows == 0:
        return f"SHORT {name} frames {n} fps {fps}"
    targets = [round((2 * w + (k + 0.5) / 4) * fps) for w in range(windows) for k in range(8)]
    sizes = {tag: (max(16, round(w0 * h / h0 / 16) * 16), h) for tag, h in HEIGHTS.items() if h0 >= h}
    writers = {}
    for tag, size in sizes.items():
        (prep / tag).mkdir(parents=True, exist_ok=True)
        wr = cv2.VideoWriter(str(prep / tag / f"{name}.tmp.avi"), cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc(*"MJPG"), 8.0, size)
        wr.set(cv2.VIDEOWRITER_PROP_QUALITY, 95); writers[tag] = wr
    pos, done = 0, 0
    for t in targets:
        while pos < t:
            if not cap.grab():
                break
            pos += 1
        ok, frame = cap.read(); pos += 1
        if not ok:
            break
        for tag, size in sizes.items():
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            writers[tag].write(small); writers[tag].write(small)
        done += 1
    cap.release()
    for wr in writers.values():
        wr.release()
    done -= done % 8
    if done == 0:
        return f"FAILED {name}"
    for tag in sizes:
        (prep / tag / f"{name}.tmp.avi").rename(prep / tag / f"{name}.avi")
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"fps": fps, "n_frames": n, "native": [w0, h0], "stored_frames": 2 * done, "windows": done // 8, "variants": {k: list(v) for k, v in sizes.items()}}))
    return f"done {name} {w0}x{h0} fps {fps:.1f} windows {done // 8} variants {sorted(sizes)}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    vids = test_videos(Path(a.root)); print(len(vids), "test videos", flush=True)
    with ProcessPoolExecutor(a.jobs) as pool:
        msgs = list(pool.map(one, [v[0] for v in vids], [str(v[1]) for v in vids], [a.root] * len(vids)))
    bad = [m for m in msgs if not m.startswith(("done", "have"))]
    print("\n".join(bad[:20])); print(f"prepared {len(msgs) - len(bad)} of {len(msgs)}; PREP_DONE", flush=True)
