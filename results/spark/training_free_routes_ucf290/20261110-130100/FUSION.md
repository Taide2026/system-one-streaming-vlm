# 1x + 2x fusion results, mix + KL, UCF-Crime test 290 (summary written 2026-11-10)

s1 = mix + KL score at 1x, s2 = score at bicubic 2x, d = s2 - s1. "Causal" = uses only this window and earlier
windows of the same video. Parameters were chosen on the 390 training videos, then applied once to test.
Intervals: paired video bootstrap, 1,000 draws, seed 20261022. Sources: tf_macro.log (this folder),
fov2-vs-fov2_2x.log (this folder), results/spark/tgm_gate_full2x_ucf290/20261110-125120/,
results/spark/vjepa_gate_online_ucf290/, results/spark/benefit_gate_ucf390/, results/spark/score_gate_ucf390/.

## Baselines

| Score | window AUC | window AP | macro AUC |
|---|---|---|---|
| all 1x | 87.9 | 47.3 | 77.9 |
| all 2x | 89.0 | 48.7 | 81.0 |

## Fusion rules (window AP all; [dAP vs all 2x]; macro AUC [vs all 2x])

| Group | Rule | Definition | AP | macro AUC |
|---|---|---|---|---|
| route per window | s1 band | use s2 when s1 is in [0.70, 0.75), else s1 (4% zoomed) | 47.6 [-2.9, +0.8] | 77.9 [-4.9, -1.4] |
| route per window | TGM area + s1 band | use s2 when TGM hot area < 0.045 and s1 in [0.70, 0.80) | 47.2 [-3.3, +0.2] | 77.8 [-5.0, -1.6] |
| route per window | TGM gate | use s2 when TGM hot area < 0.045 (80% zoomed) | 48.3 | not computed |
| route per window | V-JEPA 2 anomaly head gate | use s2 when the head fires (17.9% zoomed) | 48.5 | not computed |
| route per window | benefit head / benefit head + s1 | learned head predicts "2x helps" | 48.3 / 48.5 | not computed |
| route per window | s1-only gates | thresholds on s1 | 47.4 - 48.0 | not computed |
| route per video | running max or mean of s1 >= 0.95 | use s2 for the rest of the video | 47.3 (no window zoomed) | 77.9 |
| correct 2x | s2 - a * mean(d) of past windows, a = 0.25 | removes the video's average 2x shift | 49.1 [-0.7, +1.5] | 80.8 [-0.8, +0.4] |
| correct 2x | same, a = 0.5, current window included | | 48.8 [-1.9, +2.2] | 81.0 [-0.8, +0.8] |
| correct 2x | quantile match s2 -> s1 | map s2 to the s1 value at the same rank in past windows (5+ past windows) | 46.4 [-4.9, -0.1] | 78.0 [-4.8, -1.3] |
| correct 2x + smooth | correction a = 0.5, then mean of 2 windows | | 49.4 [-1.7, +3.1] | 81.0 [-1.6, +1.7] |
| smooth | mean of 2 windows on 2x | | 49.5 [-0.3, +1.9] | 81.5 [-1.0, +2.1] |
| smooth | EMA a = 0.85 on 2x | | 49.4 [-0.1, +1.4] | 81.2 [-0.5, +0.9] |
| smooth | mean of 2 windows on 1x | | 48.7 [-2.6, +2.3] | 78.5 [-4.7, -0.0] |
| smooth | EMA a = 0.85 on 1x | | 48.3 [-2.7, +1.8] | 78.6 [-4.3, -0.5] |
| add V-JEPA 2 | sigmoid(logit(s2) + 0.3 * V-JEPA 2 head) | | 46.4 [-5.7, +1.8] | 82.7 [+0.4, +2.9] |
| add V-JEPA 2 | sigmoid(logit(s1) + 0.3 * V-JEPA 2 head) | | 45.7 [-6.3, +0.9] | 80.2 [-2.6, +0.8] |
| normalize 1x | s1 - 0.25 * mean(s1) of past windows | | 44.2 [-9.3, +1.0] | 77.4 [-5.6, -1.8] |
| oracle (label leak) | s2 on event windows, s1 on others | | 56.7 | 86.7 |

## fov2 (8 full frames, all 2x = 49.2 AP, 79.9 macro AUC)

| Rule | AP [vs fov2 all 2x] | macro AUC |
|---|---|---|
| mean of 2 windows on 2x | 49.5 [-0.7, +1.2] | 80.4 |
| EMA a = 0.85 on 2x | 49.5 [-0.4, +0.9] | 80.4 |
