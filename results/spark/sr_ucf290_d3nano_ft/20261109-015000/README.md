# sr_ucf290_d3nano_ft / 20261109-015000

Full-frame 2x scoring of `fovfull2x_d3nano_c` (Eric, 2026-11-09): same recipe as `floor_d3nano_c` (mix + KL 0.5,
LoRA r64), trained on foveated input with frac 1.0 and upscale 2 (frames 1, 3, 5, 7 full-frame 2x).
Adapter: `results/pro6000/fovfull2x_d3nano_c/20261106-013724/` from `bear7011/streaming-vad-decision-adapters`
(revision 33f57868206bed4c78cfcd9c15e5c097b4e64916). `draft/` = original 320x240 frames, `B/` = bicubic 2x (640x480).
Scored with `run.sh` on the GB10: 18,390 windows each, B 0.713 s/window, peak GPU 4.60 GB, finished 2026-11-09T06:49Z.

## Results (window AUC / AP, %; dAP 95% interval of B vs draft)

| Model | Input | small | medium | large | all |
|---|---|---|---|---|---|
| fovfull2x_d3nano_c | draft | 84.5 / 24.8 | 90.7 / 38.1 | 91.7 / 22.5 | 88.0 / 46.8 |
| fovfull2x_d3nano_c | B 2x | 85.7 / 26.6 [-0.4, +4.2] | 92.0 / 39.0 [-2.3, +4.1] | 91.3 / 16.4 [-11.9, -0.8] | 88.8 / 46.7 [-3.1, +2.7] |
| floor_d3nano_c (20261106-142617) | draft | 83.9 / 22.5 | 90.8 / 37.2 | 92.3 / 29.0 | 87.9 / 47.3 |
| floor_d3nano_c (20261106-142617) | B 2x | 85.4 / 26.1 [+0.7, +5.9] | 92.1 / 37.9 [-2.3, +4.8] | 92.3 / 25.9 [-6.3, +0.7] | 89.0 / 48.7 [-0.8, +3.6] |

Mean score, draft -> B (`score-shift-B.log`; floor_d3nano_c values from 20261106-142617):

| Windows | fovfull2x_d3nano_c | floor_d3nano_c |
|---|---|---|
| negatives in normal videos (10,746) | 0.065 -> 0.088 | 0.071 -> 0.102 |
| negatives in anomaly videos (6,092) | 0.421 -> 0.485 | 0.427 -> 0.483 |
| small events (733) | 0.544 -> 0.617 | 0.544 -> 0.619 |
| large events (379) | 0.664 -> 0.696 | 0.688 -> 0.718 |

| File | Content |
|---|---|
| `run.sh`, `adapter-config.yaml`, `run.log` | scoring commands and log |
| `fovfull2x_d3nano_c/{draft,B}/scores.jsonl` | window scores |
| `fovfull2x_d3nano_c/eval-draft.*`, `eval-B-vs-draft.*` | `scripts/layout-eval.py` tier AUC / AP |
| `fovfull2x_d3nano_c/score-shift-B.log` | `sr_ucf290_d3nano_zs/20261106-040502/scripts/score-shift.py` |
| `scripts/shift-by-start.py`, `shift-by-start.log` | draft -> B shift of anomaly-video negatives, small and large events, inside draft-score bins (zero-shot, mix + KL, this adapter) |
