#!/usr/bin/env python3
"""Derive scripts/train-floor-tc.py from scripts/train-floor.py (2026-11-08): the streaming-vad-localization engine's
temporal_control sampling (streamvad.temporal.select: real fps + n_frames from the HIVAU items, centred sampling,
modes original / repeat4 / history4 / shuffle / static) applied to the training and validation records, exactly as
engine/scripts/train-floor.py does it (the selected indices are saved in records.jsonl / val.jsonl). It applies to every
task that reads a record's frames (v1 and c14). Config: input_form: {name: temporal_control, mode: history4, seed: 0,
history_s: 6.0}. Original file untouched."""
from pathlib import Path

src = Path("scripts/train-floor.py").read_text()
dst = Path("scripts/train-floor-tc.py")
if dst.exists():
    raise SystemExit("exists")
old = '''    write_records(train_records, out_dir / "records.jsonl")
    write_records(val_records, out_dir / "val.jsonl")
'''
assert src.count(old) == 1
# Fixed train/held-out split across training seeds (2026-11-08, review item 1): data.split_seed (default 0) drives the
# video split; data.seed keeps driving window sampling, batching and initialisation. Without this, seeds 1/2 trained on
# 121 / 119 of the 137 fixed validation videos.
_old_split = '''    train_records, val_records = sample_records(items_v,'''
assert src.count(_old_split) == 1
src = src.replace(_old_split, '''    import s1stream.traindata as _td
    _split_seed = int(dspec.get("split_seed", 0))
    _orig_split = _td.split_videos
    _td.split_videos = lambda items, holdout, seed, _s=_split_seed: _orig_split(items, holdout, _s)
    print(f"video split seed {_split_seed} (fixed); sampling seed {int(dspec.get('seed', 0))}", flush=True)
    train_records, val_records = sample_records(items_v,''')
src = src.replace(old, '''    # Apply the same temporal sampling in training, calibration and inference (streaming-vad-localization engine).
    temporal = data.get("input_form", {}) if isinstance(data.get("input_form"), dict) else {}
    if temporal.get("name") == "temporal_control":
        if int(window["frames"]) != 8:
            raise ConfigError("temporal_control requires an eight-frame budget")
        from dataclasses import replace
        from streamvad.temporal import select

        video_by_name = {v.name: v for v in items_v}

        def transform(record):
            video = video_by_name[record.video]
            indices = select(temporal["mode"], record.start_s, record.end_s, video.fps, video.n_frames,
                             record.video, record.window, int(temporal.get("seed", 0)),
                             float(temporal.get("history_s", 6.0)))
            return replace(record, frame_indices=indices)

        train_records = [transform(r) for r in train_records]
        val_records = [transform(r) for r in val_records]
        print(f"temporal_control {temporal['mode']}: indices rewritten for {len(train_records)} train + {len(val_records)} val records", flush=True)
''' + old)
dst.write_text(src)
print("wrote", dst)
