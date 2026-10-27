# probe_ucf_short_v1_f8 / run-2

Config: `config.yaml` in this folder (the base config plus the settings of this run).

Refit of the run-1 features with a slower optimizer on the GPU (2026-10-22 to 2026-10-23).
Same conditions, features and test set as run-1 (see `../run-1/README.md`).

## What changed from run-1

| Item | run-1 | run-2 |
|---|---|---|
| Optimizer | Adam lr 1e-2, 200 steps | Adam lr 1e-3, 1000 steps |
| Device | CPU | cuda (`probe-fit --device cuda`, unit probe-fit-device) |
| Loss record | none | `final_loss`, `min_loss` per mil and holmes choice in `fit.json` |
| knn | — | unchanged (no optimizer), so knn curves equal run-1 exactly |

Reason: Holmes/last of C dropped to frame AUC 0.461 at state 31 in run-1. The diagnosis in
`../run-1/diagnostics/holmes_lr.json` showed the lr 1e-2 optimizer oscillated.

## Status

| Step | Status | Output |
|---|---|---|
| Fit C | done, 2737 s, peak GPU 0.23 GiB, RSS 10 GB | `fit-C/`, `logs/fit-C.log` |
| Fit A | done, 2764 s | `fit-A/`, `logs/fit-A.log` |
| Fit B | done, 2849 s, ended 2026-10-23 02:29 | `fit-B/`, `logs/fit-B.log` |
| Compare | done, 122 s, RSS 1.8 GB | `compare.json`, `<probe>_<feature>.png`, `cka.png`, `logs/compare.log` |

Convergence: the largest (final_loss − min_loss) / min_loss over all 132 choices per condition is 0.11 % (A, mil/last/14).

**P(Yes) caveat**: the P(Yes) lines of A and B in `compare.json` and the PNG files still read the "Yes"/"No" tokens, as in
run-1. The probe numbers do not depend on P(Yes). The "A"/"B" reading is in `../run-1/pyes/pyes.json`.

## Files

| Path | Content |
|---|---|
| `compare.json` | metrics, p_yes, paired differences, layer summary, CKA (1,000 video resamples) |
| `<probe>_<feature>.png`, `cka.png` | frame AUC per state with 95 % band; linear CKA per state |
| `fit-<condition>/` | `fit.json` (with loss record), `test_scores.npy`, `test_meta.jsonl` |
| `scripts/` | `run_fit_all.sh`, `fit_gpu.py` (sets the memory fraction, calls `probe-fit --device cuda`), bench and smoke scripts |
| `diagnostics/bench_gpu_fit.json` | CPU vs GPU fit time before the run |
| `logs/` | fit, compare, smoke, work and review logs |

## Results: run-1 vs run-2 (test frame AUC)

| Curve | run-1 peak | run-2 peak | run-2 state 32 (final norm) | run-2 peak − state 32 |
|---|---|---|---|---|
| A holmes/last | 0.846 @28 | 0.841 @28 | 0.827 | 0.014 |
| B holmes/last | 0.868 @26 | 0.868 @26 | 0.834 | 0.034 |
| C holmes/last | 0.874 @23 | 0.864 @23 | 0.850 | 0.014 |
| A mil/last | 0.850 @31 | 0.850 @31 | 0.849 | 0.001 |
| B mil/last | 0.847 @31 | 0.846 @30 | 0.843 | 0.004 |
| C mil/last | 0.853 @22 | 0.853 @24 | 0.849 | 0.004 |
| A knn/last | 0.724 @31 | same | 0.655 | 0.070 |
| B knn/last | 0.741 @31 | same | 0.604 | 0.137 |
| C knn/last | 0.846 @21 | same | 0.800 | 0.045 |

- C holmes/last: the run-1 dip is gone. Lowest value over states 20–32: run-1 0.461 @31, run-2 0.838 @29.
- All other mil and holmes curves moved by at most 0.025 between run-1 and run-2.

Layer summary differences (paired bootstrap, value [95 % interval], states):

| Difference | onset | peak | peak value |
|---|---|---|---|
| C−B holmes/last | −3 [−9, +4] | −3 [−8, +7] | −0.004 [−0.030, +0.029] |
| C−B mil/last | −12 [−14, +12] | −6 [−17, +2] | +0.007 [−0.003, +0.026] |
| C−B knn/last | −11 [−14, −5] | −10 [−12, −1] | +0.105 [+0.053, +0.161] |
| A−B holmes/last | −2 [−14, +8] | +2 [−8, +7] | −0.027 [−0.061, +0.009] |
| A−B mil/last | +1 [−3, +16] | +1 [−8, +3] | +0.003 [−0.008, +0.015] |

## Reading for the research question

- With linear probes (MIL, Holmes), the decision information does not fade in the last layers of A or B:
  the drop from peak to state 32 is at most 0.034. C keeps it at the same level.
- Only kNN shows a late drop, mostly at state 32 (final norm): B −0.137, A −0.070, C −0.045.
  kNN measures the geometry without training, so this drop is in the geometry, not in the linear readability.
- C reaches its peak earlier than B. The shift is significant only for kNN (onset −11, peak −10 states).
  The MIL and Holmes intervals contain 0.
