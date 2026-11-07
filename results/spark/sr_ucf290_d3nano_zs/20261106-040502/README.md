# sr_ucf290_d3nano_zs / 20261106-040502

Full-frame super-resolution test (proposal `.specs/changes/2026-11-06-full-frame-sr/proposal.md`, local).
Frozen d3-nano, zero-shot, prompt v1, 8 full frames per window, UCF-Crime test 290 videos, 18,390 windows.

| Folder | Condition | Frame size | Tokens per frame |
|---|---|---|---|
| (draft) | saved scores `results/spark/d3_nano_ucf_test_v1_f8/20261022-012454` (main checkout) | 320x240 | 80 |
| `B/` | bicubic 2x (`cv2.INTER_CUBIC`) | 640x480 | 300 |
| `C/` | Real-ESRGAN x2plus (asset `realesrgan-x2plus`, spandrel, fp16) | 640x480 | 300 |
| `sr/` | SR config and per-video SR seconds; SR frames in `$S1_FEATURES/frames/frames_ucf290_f8_sr2/20261106-040502/` | | |

`run.sh` runs SR, then B, then C. `run.log` is its output. `seconds` in scores.jsonl = model forward only.
Smoke test (5 windows, 2026-11-06): draft reproduced exactly; SR 74.5 ms/frame; B 0.70, C 0.69 s/window; peak GPU 4.6 GB.

## Results (2026-11-06/07)

Window AUC / AP (%), tiers from `tgm_size_ucf290/20261105-200402`; [ΔAP 95 % paired bootstrap, 1,000 draws, seed 20261022].

| Input | small | medium | large | all |
|---|---|---|---|---|
| draft (original) | 76.1 / 23.1 | 87.7 / 29.5 | 89.5 / 21.7 | 82.7 / 41.9 |
| B bicubic 2x | 77.9 / 24.7 [-1.1, +4.6] | 89.1 / 29.5 [-4.7, +3.7] | 89.2 / 18.8 [-5.1, +0.3] | 83.8 / 42.6 [-1.3, +2.8] |
| B + gate 0.045 | 77.8 / 25.3 [-0.4, +5.0] | 88.7 / 28.2 [-5.5, +1.8] | 88.6 / 18.4 [-5.4, -0.8] | 83.5 / 42.1 [-1.6, +2.0] |
| C Real-ESRGAN x2 | 77.9 / 24.9 [-1.7, +5.1] | 89.7 / 32.3 [-2.3, +7.3] | 88.8 / 20.1 [-5.1, +3.3] | 83.9 / 44.0 [-1.2, +5.0] |
| C + gate 0.045 | 77.6 / 25.1 [-1.3, +4.9] | 89.4 / 30.0 [-3.6, +4.0] | 89.4 / 19.4 [-4.8, -0.1] | 83.8 / 43.3 [-1.3, +3.6] |

C vs B ΔAP: small [-1.9, +1.9], medium [-1.2, +6.3], large [-2.6, +5.8], all [-1.2, +3.4].
Gate = windows with TGM hot area >= 0.045 take the draft score; 80.0 % of windows (14,709 / 18,390) are below the gate.
Cost on the GB10: SR 64.9 ms/frame (147,120 frames, 2.65 h, peak 1.21 GB); forward B 0.704, C 0.713 s/window (draft about 0.27), peak 4.60 GB.

| Files | Content |
|---|---|
| `eval-B-vs-draft.*`, `eval-vs-draft.*` | B and C vs draft |
| `eval-C-vs-B.*` | C vs B |
| `eval-gate045.*` | gated versions |
| `score-shift-B.log`, `score-shift-C.log`, `scripts/score-shift.py` | mean / median / p99 score of normal and event windows, draft -> condition |
