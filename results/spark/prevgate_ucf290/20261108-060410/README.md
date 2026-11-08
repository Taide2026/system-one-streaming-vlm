# prevgate_ucf290: previous-window gate for the 2x bicubic up-scale (2026-11-08)

Question: does up-scaling only the windows after a suspicious window keep the small-tier gain without the large-tier loss?
Offline, CPU only. It reuses the saved per-window scores of the original input and the full-frame bicubic 2x input.

- Rule: window t uses the 2x score when the score used at window t-1 is in [lo, hi]. The rule is causal: when
  window t-1 was up-scaled, its 2x score is the one used. The first window of each video keeps the original score.
- Gates: [0.3, 0.7] (the default band of the `foveated_gated` input form), p >= 0.1, 0.2, 0.3, 0.5.
  The thresholds were not tuned on held-out data: they are read on the test set.
- Models: zero-shot d3-nano and `floor_d3nano_c` (mix + KL). Inputs: `config.yaml`.
- Metrics: `scripts/layout-eval.py` (window AUC / AP per tier, 1,000-draw paired video bootstrap, seed 20261022).

Also `decompose.py` → `decompose.log`: AP per tier when event windows and negative windows each take the original or the 2x score.

Files: `gate.py`, `config.yaml`, `run.sh`, `run.log`, `eval-zeroshot.json`, `eval-mixkl.json`, `rates.json`
(share of windows up-scaled per window group).
