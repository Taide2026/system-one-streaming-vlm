# Oracle split of mix + KL (floor_d3nano_c) scores, test 290

Made 2026-11-09. No new model run: `split.py` mixes two existing score files.

- 1x scores: `results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c/draft/scores.jsonl`
- 2x scores (bicubic): `results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c/B/scores.jsonl`
- `oracle.jsonl`: event windows (label 1) take the 2x score, negative windows take the 1x score. Uses labels, so not a usable method.
- `negonly.jsonl`: event windows 1x, negative windows 2x.
- `eval.json`, `eval.log`: `scripts/layout-eval.py`, paired video bootstrap 1,000 draws against the 1x scores.

| Layout | small | medium | large | all |
|---|---|---|---|---|
| oracle | 32.8 [+5.9, +13.4] | 46.4 [+6.3, +12.2] | 34.7 [+2.2, +7.3] | 56.7 [+7.4, +11.8] |
| negatives only 2x | 17.1 [-7.8, -2.7] | 29.9 [-9.8, -3.8] | 21.2 [-11.1, -2.3] | 39.6 [-9.3, -5.7] |
| all 2x | 26.1 [+0.7, +5.9] | 37.9 [-2.3, +4.8] | 25.9 [-6.3, +0.7] | 48.7 [-0.8, +3.6] |
