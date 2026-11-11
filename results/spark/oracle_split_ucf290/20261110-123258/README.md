# Oracle split of floor2x_hn_d3nano_c scores, test 290

Made 2026-11-10 (Eric: "先把oracle也算出來"). No new model run: `split.py` (copy of 20261109-151315/split.py) mixes
the floor2x_hn 1x and 2x scores of `results/spark/sr_ucf290_d3nano_ft/20261110-023316/{draft,B}/scores.jsonl`.

- `oracle.jsonl`: events 2x, negatives 1x (uses labels, not a usable method). `negonly.jsonl`: events 1x, negatives 2x.
- `eval-vs-mixkl.*`: scripts/layout-eval.py against mix + KL 1x. `eval-vs-hn1x.*`: against floor2x_hn 1x.

| Layout (window AUC / AP %) | small | medium | large | all |
|---|---|---|---|---|
| hn oracle, ΔAP vs mix + KL 1x | 85.3 / 27.9 [-1.4, +12.7] | 92.0 / 32.9 [-12.2, +4.8] | 91.8 / 26.2 [-11.0, +6.9] | 88.8 / 48.3 [-3.9, +5.9] |
| hn oracle, ΔAP vs hn 1x | [+0.5, +4.7] | [-2.4, +5.2] | [-1.3, +2.7] | [+0.5, +3.8] |
| hn negatives only 2x, vs mix + KL 1x | 84.2 / 24.9 [-3.0, +10.2] | 91.3 / 30.6 [-15.8, +2.5] | 92.1 / 24.4 [-12.8, +5.1] | 88.1 / 45.1 [-7.4, +3.3] |
| hn negatives only 2x, vs hn 1x | [-2.2, +2.3] | [-5.4, +3.0] | [-4.6, +3.2] | [-3.4, +2.2] |

One seed. Bootstrap: 1,000 paired video draws, seed 20261022, no multiple-comparison correction.
