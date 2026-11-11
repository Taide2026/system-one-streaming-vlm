# fovfull2x_d3nano_c scored in its own training format (2026-11-09)

Question: does fovfull2x_d3nano_c lose large-tier AP at 2x only because the test input differs from its training input?

- Input `odd-bicubic2`: frames 1, 3, 5, 7 at bicubic 2x, frames 0, 2, 4, 6 at 1x. Equal to the training input
  `foveate(frac=1.0, upscale=2)` (tests/test_sr_score_odd.py).
- Adapter: `results/pro6000/fovfull2x_d3nano_c/20261106-013724`. Test: UCF 290 videos, 18,390 windows.
- Run: `run.sh`, log `run.log`. 0.477 s/window, 8765 s, peak GPU 4.60 GB.
- draft (1x) and B (all frames 2x) scores come from `../20261109-015000/fovfull2x_d3nano_c/`.

## Window AUC / AP

| Input | small | medium | large | all |
|---|---|---|---|---|
| draft (1x) | 84.5 / 24.8 | 90.7 / 38.1 | 91.7 / 22.5 | 88.0 / 46.8 |
| odd-bicubic2 | 85.9 / 26.3 | 91.5 / 39.2 | 91.0 / 16.8 | 88.7 / 46.5 |
| B (all 2x) | 85.7 / 26.6 | 92.0 / 39.0 | 91.3 / 16.4 | 88.8 / 46.7 |

ΔAP 95% interval, odd-bicubic2 vs draft: small [-0.2, +3.2], medium [-1.9, +3.7], large [-11.0, -1.0], all [-2.7, +2.0].

## Files

| File | Content |
|---|---|
| `fovfull2x_d3nano_c/odd/scores.jsonl` | scores, one line per window |
| `fovfull2x_d3nano_c/eval-odd-vs-draft.*` | `scripts/layout-eval.py` tier AUC / AP (odd and B vs draft) |
| `fovfull2x_d3nano_c/score-shift-odd.log` | mean score draft -> odd per group (`sr_ucf290_d3nano_zs/20261106-040502/scripts/score-shift.py`) |
| `eval-variants-vs-mixkl.json` | `scripts/layout-eval.py`: mix + KL 2x, fov2 2x, fovfull2x 1x / odd / 2x, each against mix + KL 1x (floor_d3nano_c draft), paired video bootstrap |
