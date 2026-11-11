# floor2x_hn_d3nano_c on UCF-Crime test 290, 1x and 2x (2026-11-10)

Adapter: `results/spark/floor2x_hn_d3nano_c/20261109-134518` (GB10, 626 steps, T = 1.18).
`run.sh` waited for the training, ran a 5-window smoke test (`results/smoke/floor2x_hn_ft/`), then scored
draft (1x, 4715 s, 0.253 s/window) and B (bicubic 2x, 12991 s, 0.701 s/window). Peak GPU 4.60 GB.

## Window AUC / AP (%), ΔAP 95% interval against mix + KL 1x (floor_d3nano_c draft)

| Input | small | medium | large | all |
|---|---|---|---|---|
| mix + KL 1x | 83.9 / 22.5 | 90.8 / 37.2 | 92.3 / 29.0 | 87.9 / 47.3 |
| mix + KL 2x | 85.4 / 26.1 [+0.7, +5.9] | 92.1 / 37.9 [-2.3, +4.8] | 92.3 / 25.9 [-6.3, +0.7] | 89.0 / 48.7 [-0.8, +3.6] |
| floor2x_hn 1x | 83.4 / 25.1 [-3.1, +10.0] | 91.1 / 32.1 [-13.2, +2.7] | 92.0 / 25.3 [-12.0, +5.7] | 87.7 / 46.0 [-5.6, +3.6] |
| floor2x_hn 2x | 85.9 / 27.3 [-1.5, +12.4] | 92.3 / 31.4 [-14.6, +4.1] | 91.9 / 25.0 [-12.4, +6.2] | 89.2 / 47.3 [-5.5, +5.6] |

floor2x_hn 2x against its own 1x: ΔAP small [-1.1, +5.1], medium [-5.4, +4.7], large [-4.5, +4.4], all [-1.8, +4.4].

## Success criteria (from the training design)

| Criterion | mix + KL | floor2x_hn |
|---|---|---|
| Anomaly-video negative mean, 1x -> 2x | 0.427 -> 0.483 (+0.056) | 0.338 -> 0.330 (-0.008) |
| Large AP, 1x -> 2x | 29.0 -> 25.9 (-3.1) | 25.3 -> 25.0 (-0.3) |

## Frame AUC / AP (%), Δ 95% interval against mix + KL 1x

| Input | frame AUC | frame AP |
|---|---|---|
| mix + KL 1x | 88.06 | 45.16 |
| mix + KL 2x | 89.11 [+0.07, +2.20] | 46.54 [-0.79, +3.66] |
| floor2x_hn 1x | 87.78 [-1.62, +1.09] | 43.97 [-5.82, +4.15] |
| floor2x_hn 2x | 89.27 [-0.28, +2.75] | 45.28 [-5.46, +6.07] |

## Files

`draft/`, `B/` scores; `eval-vs-mixkl.*`, `eval-2x-vs-1x.*`, `eval-1x-vs-mixkl.*` (scripts/layout-eval.py);
`score-shift-B.log`; `frame-auc.*` (results/spark/frame_auc_ucf290/20261109-153701/frame-auc.py); `run.log`.
One seed. Bootstrap: 1,000 draws, seed 20261022, no multiple-comparison correction.
