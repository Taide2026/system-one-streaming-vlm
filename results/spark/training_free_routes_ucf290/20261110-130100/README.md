# Training-free routers and score transforms, mix + KL, test 290 (2026-11-10)

Eric asked for ways to beat all-2x without fine-tuning. Fable (subagent, read-only) tested 13 causal rules with
parameters chosen on the 390 training videos (train scores: results/spark/train_zoom_scores_ucf390/20261108-131218/).
`tf.py` is its script (copied from the session scratchpad). Its full table is in the session; main result below.

Verified here: causal 2-window mean (score = mean of this window and the previous one of the same video),
`smooth2_1x.jsonl`, `smooth2_2x.jsonl`, `eval-vs-mix1x.*` (scripts/layout-eval.py, 1,000 draws, seed 20261022).

| Rule | small | medium | large | all |
|---|---|---|---|---|
| mean of 2 windows, 1x | 24.1 [-0.1, +2.9] | 39.9 [-0.1, +4.7] | 31.6 [-1.1, +4.7] | 48.7 [+0.3, +2.3] |
| mean of 2 windows, 2x | 27.3 [+0.9, +7.9] | 40.0 [-1.0, +6.5] | 26.8 [-5.4, +1.7] | 49.5 [-0.2, +4.7] |

Fable: smoothed 2x vs all-2x ΔAP all [-0.3, +1.9]. No rule chosen on train beat all-2x significantly.
