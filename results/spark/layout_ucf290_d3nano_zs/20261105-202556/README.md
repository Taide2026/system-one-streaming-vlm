# layout_ucf290_d3nano_zs

Frame-layout test on the frozen d3-nano (zero-shot, prompt v1), UCF-Crime test 290 videos, 18,390 windows.
Script: `scripts/layout-test.py`. Frames: `$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529` (from `scripts/extract-frames.py`).
Crop: TGM 1/3 box (`box_third` in `results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl`), 2x cubic up-scale. No area gate.

| Folder | Layout | Images |
|---|---|---|
| `A/` | F0 C1 F2 C3 F4 C5 F6 C7 (interleaved, fov1 format) | 8 |
| `B/` | F0 C0 F2 C2 F4 C4 F6 C6 (paired, 4 time points) | 8 |
| `C/` | F0 C0 F1 C1 ... F7 C7 (paired, 8 time points) | 16 |

Draft (8 full frames) = `results/spark/d3_nano_ucf_test_v1_f8/20261022-012454` in the main checkout; the draft layout of
the script reproduced its first 5 scores exactly. Tiers: `results/spark/tgm_size_ucf290/20261105-200402/jev-hard-events.jsonl`.
Each folder holds `scores.jsonl` and `run.log`. `manifest.json` is the frames run manifest.
