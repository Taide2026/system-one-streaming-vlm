# Frame-level AUC with paired video bootstrap, UCF-Crime test 290

Made 2026-11-09. No new model run: `frame-auc.py` reads existing window scores.

- Each frame takes the max score of the windows that cover it (`s1stream.metrics.frame_scores`). Frames no window covers are left out.
- Bootstrap: 1,000 draws, seed 20261022, anomaly (140) and normal (150) videos drawn separately, the same draws for every run.
- For each of AUC and AP: `delta_ci` is the paired 95% interval against `mix1x` (mix + KL, floor_d3nano_c, 1x); `ci` is the interval of the value itself.
- `run.sh` lists the score files. `run.log` holds the printed table; `frame-auc.json` the numbers.
- Second attempt stopped by hand to add frame AP; the third run computes AUC and AP.
- First attempt failed: the manifest `events` field is already a list, not a JSON string. Fixed and rerun in the same folder.
