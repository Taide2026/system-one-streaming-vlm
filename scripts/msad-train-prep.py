#!/usr/bin/env python3
"""MSAD Protocol-ii TRAINING videos at 180p for in-domain fine-tuning (2026-11-14). CPU only.

Every frame of every training-list video, area-downscaled to height 180 (width a multiple of 16, as msad-prep.py does
for the test videos), Motion-JPEG quality 95 at the native fps: <root>/prep/train180/<name>.avi and a <name>.mp4 link
to it (the trainer looks for .mp4). Writes <root>/prep/msad_train.json in the HIVAU layout the trainer reads:
  {name: {n_frames, fps, label: [type] | ["Normal"], events: [[start_s, end_s]]}}
Events from anomaly_annotation.csv (native frames, inclusive). Refuses to run if a training name is in the test list.
"""
import argparse, csv, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2


def names(root, f):
    return [Path(l).name.replace("_i3d.npy", "") for l in (root / f).read_text().split()]


def src(root, name):
    if name.startswith("MSAD_normal_"):
        stem = name[len("MSAD_normal_"):]      # folders may hold a space ("pedestrian street"), the files do not
        hits = sorted((root / "MSAD_normal_training").glob(f"*/{stem}.mp4"))
        return hits[0] if hits else root / "MSAD_normal_training" / f"{stem}.mp4"
    return root / "MSAD_anomaly" / name.rsplit("_", 1)[0] / f"{name}.mp4"


def one(name, path, out):
    out = Path(out); avi = out / f"{name}.avi"; meta = out / f"{name}.json"
    if meta.is_file():
        return json.loads(meta.read_text())
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"name": name, "error": f"cannot open {path}"}
    fps, w0, h0 = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if h0 < 180 or fps <= 0:
        return {"name": name, "error": f"source {w0}x{h0} fps {fps}"}
    size = (max(16, round(w0 * 180 / h0 / 16) * 16), 180)
    tmp = out / f"{name}.tmp.avi"
    wr = cv2.VideoWriter(str(tmp), cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc(*"MJPG"), fps, size); wr.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        wr.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA)); n += 1
    cap.release(); wr.release()
    if n < 2 * fps:
        tmp.unlink(missing_ok=True); return {"name": name, "error": f"short: {n} frames"}
    tmp.rename(avi)
    link = out / f"{name}.mp4"
    if not link.exists():
        os.symlink(avi.name, link)
    m = {"name": name, "fps": fps, "n_frames": n, "native": [w0, h0]}
    meta.write_text(json.dumps(m)); return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("root", type=Path); ap.add_argument("--jobs", type=int, default=24); a = ap.parse_args()
    train, test = names(a.root, "MSAD_I3D_WS_Train.list"), set(names(a.root, "MSAD_I3D_WS_Test.list"))
    assert not set(train) & test, f"train/test overlap: {sorted(set(train) & test)[:5]}"
    out = a.root / "prep" / "train180"; out.mkdir(parents=True, exist_ok=True)
    spans = {}
    for row in csv.reader((a.root / "anomaly_annotation.csv").open()):
        if row and row[0] != "name" and len(row) >= 5 and row[3].strip() and row[4].strip():
            spans[row[0].strip()] = (int(float(row[3])), int(float(row[4])))
    with ProcessPoolExecutor(a.jobs) as pool:
        metas = list(pool.map(one, train, [str(src(a.root, n)) for n in train], [str(out)] * len(train)))
    bad = [m for m in metas if "error" in m]; db = {}
    for m in metas:
        if "error" in m:
            continue
        n = m["name"]
        if n.startswith("MSAD_normal_"):
            db[n] = {"n_frames": m["n_frames"], "fps": m["fps"], "label": ["Normal"], "events": []}
        elif n in spans:
            s, e = spans[n]; e = min(e, m["n_frames"] - 1)
            db[n] = {"n_frames": m["n_frames"], "fps": m["fps"], "label": [n.rsplit("_", 1)[0]], "events": [[s / m["fps"], (e + 1) / m["fps"]]]}
        else:
            bad.append({"name": n, "error": "anomaly video without annotation"})
    (a.root / "prep" / "msad_train.json").write_text(json.dumps(db))
    print("\n".join(json.dumps(b) for b in bad[:20]))
    print(f"train list {len(train)}; prepared {len(db)} ({sum(1 for v in db.values() if v['label'] != ['Normal'])} abnormal); skipped {len(bad)}; TRAINPREP_DONE", flush=True)
