# TGM-area gate for full-frame 2x, mix + KL, test 290

Made 2026-11-10 (Eric: "用tgm篩出small然後只把small放大"). CPU only, no model run.
A window takes its mix + KL 2x score when its TGM hot_area < 0.045 (fov2's max_area, fixed before the run), else 1x.
Scores: results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c/{draft,B}/scores.jsonl.

| File | Content |
|---|---|
| gate.py, gate.log, gated.jsonl | Gate, zoom rate per group |
| tier_rates.py, tier_rates.log | Zoom rate of event windows per tier |
| eval-vs-mix1x.*, eval-vs-mix2x.* | scripts/layout-eval.py against mix + KL 1x and all 2x |

Zoom rate: events 70.3% (small 91.8%, medium 73.6%, large 24.8%), anomaly-video negatives 79.8%,
normal-video negatives 81.5%, all windows 80.0%.

| Window AUC / AP % | small | medium | large | all |
|---|---|---|---|---|
| gate, ΔAP vs mix 1x | 85.7 / 27.0 [+1.6, +6.7] | 91.5 / 36.7 [-3.1, +2.5] | 91.4 / 24.3 [-7.8, -0.8] | 88.8 / 48.3 [-1.0, +3.1] |
| gate, ΔAP vs mix all 2x | [-0.0, +1.8] | [-2.8, +0.2] | [-3.2, -0.1] | [-1.3, +0.5] |

Bootstrap: 1,000 paired video draws, seed 20261022, no multiple-comparison correction.
