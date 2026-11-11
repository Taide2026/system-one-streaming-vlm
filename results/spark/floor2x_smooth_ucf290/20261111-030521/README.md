# Causal smoothing of floor2x_d3nano_c scores, UCF-Crime test 290 (2026-11-11)

smooth.py: mean2 = mean of this and the previous window; ema85 = EMA a 0.85 (same rules as training_free_routes_ucf290/20261110-130100/tf.py; mean2 on mix + KL 2x reproduces smooth2_2x.jsonl exactly).
Files: eval-vs-mix1x.log (window AUC/AP vs mix + KL 1x), eval-vs-f2x2x.log (vs floor2x 2x raw), eval-vs-mix2x-mean2.log, macro.log, frame-auc.log.
