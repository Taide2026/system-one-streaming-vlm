# probe_ucf_short_v1_f8 / run-1

Config: `config.yaml` in this folder (the base config plus the settings of this run).

Layer probing of three conditions on UCF-Crime short (2026-10-22). Config: `configs/probe_ucf_short_v1_f8.yaml`.

| Condition | Weights | Attention |
|---|---|---|
| A | Qwen3.5-9B | causal |
| B | Qwen3.5-9B | 8 full-attention layers non-causal |
| C | d3-flash | 8 full-attention layers non-causal |

Test set: 117 videos (49 anomaly, 68 normal), 1,684 windows of 2 s, 197 positive windows.

## Status

| Step | Status | Output |
|---|---|---|
| Extract (`probe-extract`) | done, commit 692144a | features in `$S1_FEATURES/probe_ucf_short_v1_f8/<condition>/run-1/` (not in git); copies of `run.json` in `extract/` |
| Fit (`probe-fit`) | done | `fit-A/`, `fit-B/`, `fit-C/` |
| Compare (`probe-compare`) | **preview**, commit 680a2ea | `preview-compare/` |
| P(Yes) reference with code tokens | **preview**, scratch scripts | `pyes/pyes.json` |

The compare output is a preview: the P(Yes) lines of A and B in `preview-compare/` read the "Yes"/"No" tokens.
The prompt asks for the option code, so the matching reading is the "A"/"B" tokens (`head_AB` in `pyes/pyes.json`).
The probe numbers do not depend on P(Yes). The official `compare.json` comes after the P(Yes) fix.

## Files

| Path | Content |
|---|---|
| `windows.csv` | One row per test window: video, time, label, P(Yes) of A, B, C ("A"/"B" code tokens), and each probe's score at its peak state (column name ends with the state). Peak states are chosen on the test set. No threshold is applied |
| `val_selected.json` | One state per condition, probe and feature, picked by the validation score in `fit.json` (MIL: lowest video BCE; Holmes, kNN: highest AUC), with its test frame AUC and interval. This is the fair single-layer result |
| `curves.csv` | Every probe curve: condition, probe, feature, state, frame AUC with 95 % interval, frame AP, window AUC, linear CKA against B |
| `preview-compare/compare.json` | Full compare output: metrics, p_yes, paired differences, layer summary (onset, peak, peak value), CKA |
| `preview-compare/<probe>_<feature>.png` | Frame AUC per state, three conditions, 95 % band, P(Yes) dashed |
| `preview-compare/cka.png` | Linear CKA per state for A-B, C-B, C-A |
| `pyes/pyes.json` | Test P(Yes) per window for four readings (`stored`, `head_AB`, `head_YesNo`, `readout_AB`), with frame AUC and window AUC |
| `fit-<condition>/` | `fit.json` (chosen settings, validation scores), `test_scores.npy` (3 probes, 2 features, 33 states, 1,684 windows), `test_meta.jsonl` |
| `extract/<condition>.run.json` | Extraction record: times, peak GPU, asset revisions, `visual_weights_equal` |
| `logs/` | Smoke, extraction, fit and compare logs |
| `scripts/` | Scratch scripts used for the checks and the P(Yes) preview. Not part of the package |

## Peak frame AUC on the test set (preview)

Best state per curve, 95 % bootstrap interval over 1,000 video resamples.

| Probe / feature | A | B | C |
|---|---|---|---|
| MIL / last | 0.850 [0.802, 0.893] @31 | 0.847 [0.799, 0.892] @31 | 0.853 [0.812, 0.893] @22 |
| MIL / visual | 0.847 [0.805, 0.888] @21 | 0.851 [0.811, 0.892] @19 | 0.856 [0.817, 0.896] @18 |
| Holmes / last | 0.846 [0.788, 0.895] @28 | 0.868 [0.825, 0.904] @26 | 0.874 [0.834, 0.909] @23 |
| Holmes / visual | 0.815 [0.761, 0.865] @16 | 0.808 [0.749, 0.865] @32 | 0.804 [0.737, 0.861] @19 |
| kNN / last | 0.724 [0.652, 0.794] @31 | 0.741 [0.664, 0.814] @31 | 0.846 [0.799, 0.888] @21 |
| kNN / visual | 0.756 [0.680, 0.824] @22 | 0.752 [0.673, 0.818] @22 | 0.789 [0.724, 0.850] @22 |

The best state is chosen on the test set, so these peak values are slightly optimistic.

## P(Yes) frame AUC on the test set

| Condition | "A"/"B" code tokens | "Yes"/"No" tokens |
|---|---|---|
| A | 0.835 | 0.847 |
| B | 0.848 | 0.839 |
| C (d3-flash readout, unchanged) | 0.860 | 0.852 |

## Layer shift B -> C (fine-tune effect)

Paired bootstrap difference C - B; a change counts only when the interval excludes 0 (decision 12).

| Probe / feature | Onset | Peak | Peak value |
|---|---|---|---|
| kNN / last | -11 [-14, -5] | -10 [-12, -1] | +0.105 [+0.053, +0.161] |
| kNN / visual | -1 [-4, 3] | 0 [-5, 7] | +0.037 [+0.009, +0.073] |
| MIL / last | -12 [-14, 10] | -9 [-17, 1] | +0.006 [-0.004, 0.025] |
| MIL / visual | 1 [-2, 7] | -1 [-13, 6] | +0.005 [-0.006, 0.012] |
| Holmes / last | -1 [-7, 4] | -3 [-8, 7] | +0.006 [-0.017, 0.034] |
| Holmes / visual | -2 [-11, 9] | -13 [-28, 17] | -0.005 [-0.033, 0.017] |

A -> B (attention effect): no onset, peak or peak-value difference excludes 0.

## Earlier d3-flash runs on the same 117 videos

| Run | Question | Frames | Frame AUC | Window AUC |
|---|---|---|---|---|
| `results/d3_flash_ucf_short_f8/20261016-133125` | "Is something abnormal happening?" | 8 | 0.850 | 0.843 |
| `results/d3_flash_ucf_short_f16/20261016-135410` | same | 16 | 0.853 | 0.845 |
| `results/d3_flash_ucf_short_c14_f8/20261020-003356` | "What is happening?" (14 classes) | 8 | 0.870 | 0.862 |
| this run, condition C | v1 (MoniTor) | 8 | 0.860 | 0.852 |

## Known issues

- The regularization weight lambda is at the top of the grid (0.01) for most MIL and Holmes states; the grid may be too narrow.
- Holmes / last of C drops to 0.727 at state 24 (neighbours about 0.86); not investigated.
- CKA has no bootstrap interval.
