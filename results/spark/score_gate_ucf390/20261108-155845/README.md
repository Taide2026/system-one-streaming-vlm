# score_gate_ucf390 / 20261108-155845

Score-based gates and score fusion for the 2x cascade. CPU only. GB10, 2026-11-09 (UTC 2026-11-08 15:58).
Eric approved trying the cheap ideas of the Sol survey `~/Eric/notes/raw/sessions/2026-11-08-gate-selection-survey-sol.md`.

Question: can the VLM's own 1x score (band, causal rise), TGM motion, or fusing 1x and 2x scores pick or score the
zoomed windows better than the anomaly-head online gate (mix+KL AP all 48.5)?

| File | Content |
|---|---|
| `config.yaml` | Paths, grids, main rules (fixed before the run) |
| `gates.py` | Gates, fusion, selection on train, test evaluation |
| `run.log`, `gates.json` | Train selection, test AUC/AP per tier with 95% bootstrap CI, test zoom rates |

    timeout 3600 uv run --frozen python gates.py --config config.yaml > run.log 2>&1   # from this folder

All cutoffs (20% of the training windows) and parameters (band center, motion beta, fusion) come from the 390
training videos, in-sample. The anomaly-head gate uses c = 1.095 from `vjepa_gate_online_ucf290/20261108-115929`.
Its replacement row reproduces that run (zero-shot 42.8, mix+KL 48.5 AP all).

## Results (AP all, test 290 videos; actual test zoom rate)

| Gate | zero-shot replace | zero-shot main fusion | mix+KL replace | mix+KL main fusion |
|---|---|---|---|---|
| none / all 2x / oracle | 41.9 / 42.6 / 47.4 | | 47.3 / 48.7 / 56.7 | |
| band (center 0.65 zs, 0.8 mixkl) | 41.9 (15.1%) | 42.3 (lambda 1, delta) | 47.5 (11.1%) | 47.5 (max) |
| band center 0.7 (Sol, reference) | 42.3 (13.9%) | 42.4 | 47.4 (12.9%) | 47.9 (lambda 0.5, delta) |
| rise (= rise + motion, beta picked 0) | 42.1 (15.0%) | 42.4 (lambda 1, delta) | 48.0 (16.6%) | 47.4 (max) |
| anomaly head online | 42.8 (17.9%) | 43.2 (max), CI [+0.4, +2.5] | 48.5 (17.9%) | 48.5 (max) |
| fusion on all windows (reference) | | 43.2 (lambda 0.75) | | 48.1 (lambda 0.25) |

delta = mean(s2 - s1) over the training negatives: zero-shot +0.0072, mix+KL +0.0308.
