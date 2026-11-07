# Frame reading and runner timing: change record (2026-11-07)

Two changes from the codebase scan, made on `experiment/small-event-fov` only. Not pushed.

| Commit | Change |
|---|---|
| `bd68c57` `perf(frames): read window frames sequentially` | `read_frames` reads forward instead of one seek per frame |
| `05252bc` `feat(runner): record input and model seconds per window` | `scores.jsonl` gets `input_seconds` and `model_seconds` |

Base before the changes: `285f87f`. See the full diff:

```
git diff 285f87f 05252bc -- src tests
git show bd68c57 -- src/s1stream/input_forms/frames.py
git show 05252bc -- src/s1stream/runner.py
```

## 1. `read_frames` (src/s1stream/input_forms/frames.py)

The signature `read_frames(path, indices) -> list[PIL.Image]` does not change. Callers need no change.

| | Before | After |
|---|---|---|
| Open the video | On every call, released at the end | Once per thread; it stays open until the thread reads another video |
| Go to a frame | `cap.set(CAP_PROP_POS_FRAMES, index)` for every frame | `cap.grab()` forward to the frame |
| Seek | Every frame | Only on a backward jump or a forward gap above `MAX_GRAB` = 100 frames |
| Same index twice | Seek and decode again | The last decoded frame is returned again |
| Failed read | `FrameReadError` | `FrameReadError`; the reader closes, so the next call reopens the video |
| Threads | One capture per call | One reader per thread (`threading.local`): prefetch threads do not share a capture |
| Forked child (DataLoader worker) | Nothing open, nothing inherited | The inherited reader is set aside; the child opens its own (see section 3) |

Why `MAX_GRAB` = 100: on UCF-Crime 320x240 H.264, one `grab()` takes 0.08-0.16 ms and one seek + read takes 15-17 ms.
So 100 grabs cost about one seek.

Callers that get the change:

| Caller | Access pattern | Effect |
|---|---|---|
| `runner` with input form `frames` (`Frames.build`) | windows in order | sequential, fastest case |
| `foveated`, `foveated_gated` input forms | windows in order | sequential |
| `src/s1stream/train.py` `WindowDataset` | random windows | one seek per window, then grabs |
| `scripts/train-floor.py` | random windows | one seek per window, then grabs |
| `scripts/extract-frames.py` | windows in order | sequential |
| `scripts/score-records.py`, `mine-hardneg.py`, `head-features.py` | the order of their input (not checked) | sequential when the windows of one video come in order; else one seek per window |

## 2. Runner timing (src/s1stream/runner.py)

| Field in `scores.jsonl` | Before | After |
|---|---|---|
| `seconds` | input form + backbone + head (prefetch: backbone + head) | unchanged |
| `input_seconds` | not written | wall time of `input_form.build` (frame reading). With prefetch it is measured in the background thread |
| `model_seconds` | not written | wall time of backbone + head, after a CUDA synchronize |

- New helper `timed_build(input_form, item, window, question)` returns `(input, seconds)`. The sequential path and the prefetch threads both use it.
- `report.py` still reads `seconds`. Old run folders have no new fields and still work.
- Sequential runs: `seconds` >= `input_seconds` + `model_seconds`. The difference is the context and question setup.

## 3. Problem found during the change: fork with an open video

With the old code, no video stayed open, so a forked child inherited nothing.
With the new code, the parent can hold an open `VideoCapture` when it forks DataLoader workers.
Test on the GB10, `forkhang.py` in the session scratchpad, not in git:

| What the child does with the inherited capture | Result |
|---|---|
| Does not touch it, opens a new capture | works |
| Calls `release()` | hangs |
| Reads from it | hangs |

The parent process had 58 threads with the open capture (FFmpeg decoder threads). These threads do not exist in the child.
Fix: `os.register_at_fork(after_in_child=...)` moves the inherited reader to a module list. The list keeps the object alive, so its destructor never runs.
The training code reads frames inside DataLoader workers (6 for train, 2 for val). So the parent holds a video only with `workers: 0`.

## 4. Verification

### Unit tests

```
uv run --frozen pytest tests/test_frames.py tests/test_runner.py tests/test_ucf.py -q
39 passed in 2.32s
```

- `tests/test_frames.py` (new, 8 tests) uses a synthetic 150-frame MJPG video. Each frame has its own grey level. The old seek-per-frame reader is the reference. The tests cover:
  - consecutive windows;
  - repeated indices;
  - a backward jump inside one call and between calls;
  - a large gap (seek count checked);
  - two videos interleaved;
  - a read past the end (then the next read works);
  - 4 threads on one video;
  - a forked child.
- The fork test fails when the guard is removed (checked).
- `tests/test_runner.py` adds 3 assertions and 2 tests:
  - slow fake plugins (input 0.05 s, model 0.01 s): `input_seconds` >= 0.05 and 0.01 <= `model_seconds` < 0.05;
  - prefetch: `input_seconds` >= 0.05 and `seconds` < 0.05.

### Real data: new reader against the saved frames of the old reader

Command: `run.sh` (CPU only, `timeout 1800`). Script: `check.py`. One line per video and pass: `check-log.jsonl`. Totals: `summary.json`.
Reference: `$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529/`, written by `scripts/extract-frames.py` at `e0c2a1b` with the old `read_frames`.
Windows: `config.yaml` (2.0 s window, 2.0 s stride, 8 frames), as in the frames run.

| Pass | Videos | Windows | Windows that differ | Max pixel difference | ms per window |
|---|---|---|---|---|---|
| New reader, windows in order (runner pattern) | 290 | 18,390 | 0 | 0 | 8.0 |
| New reader, windows shuffled (training pattern) | 30 | 3,065 | 0 | 0 | 25.1 |
| Old reader, windows in order | 10 | 420 | 0 | 0 | 152.0 |

On the same 10 videos (420 windows): old reader 152.0 ms per window, new reader 7.7 ms per window.
The old reader took 115-119 ms per window in the earlier measurement of the codebase scan, on the same machine. The load on the machine was different.
The ms per window covers `read_frames` only, not the comparison with the saved frames.

## 5. Not checked

- No model run with the new runner. The new fields are checked with fake plugins only.
- The `recompute` context strategy with real video: each window jumps back to the past frames, so it seeks once per window.
- XD-Violence videos (another source and other encodings). Only UCF-Crime and the synthetic MJPG video were checked.
