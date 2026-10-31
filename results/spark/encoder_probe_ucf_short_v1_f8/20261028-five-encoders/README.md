# 20261028-five-encoders: probe 5 frozen vision encoders on the probe_ucf_short_v1_f8 windows (2026-10-28)

This folder was named `run-1` until 2026-10-31. The feature folders in `$S1_FEATURES` keep the run name `run-1`,
and the logs and `fit.json` files still show the old folder name.

Proposal: encoder-probing (local `.specs/changes/2026-10-28-encoder-probing/proposal.md`). Config: `config.yaml`.
Smoke test and decision 9 check on 5 windows: `../smoke-1/`.

## Steps

1. `scripts/run.sh` (worktree root): `s1stream encoder-extract` for each encoder, one at a time; features in
   `$S1_FEATURES/encoder_probe_ucf_short_v1_f8/<folder>/run-1/`. Logs: `logs/extract-<encoder>.log`, `logs/run.log`.
2. `run.sh` then ran the decision 9 check on all `qwen35-vit` windows against vit-1 (`../smoke-1/scripts/check_equal.py run-1`).
   It failed (`logs/check-equal.log`) and `run.sh` stopped there (`set -e`).
3. Diagnosis (below). Eric chose to go on (2026-10-28).
4. `scripts/fit.sh`: `s1stream probe-fit --device cuda` for each of the 7 feature folders (`fit-<folder>/`, `logs/fit-<folder>.log`),
   then `s1stream encoder-compare`: `encoder_compare.json`, `mil.png`, `holmes.png`, `knn.png` (`logs/compare.log`).

| Step | Start (UTC) | End (UTC) |
|---|---|---|
| Extract, 5 encoders, 12,772 windows each | 2026-10-28 07:19 | 12:04 |
| Fit, 7 folders | 13:18 | 15:50 |
| Compare | 15:50 | 15:51 |

Free memory before every step: 106 to 107 GiB.

## Decision 9 check: failed, cause found

`logs/check-equal.log`: all 6 (set, kind) pairs fail the rule "cosine >= 0.9999 for every window and state".

- `scripts/check_equal_spread.py` (`logs/check-equal-spread.log`): 0.12 to 0.17 % of the (window, state) pairs of
  `patch_max` and at most 0.011 % of `patch_mean` are below 0.9999; minimum 0.99835; median 0.999994 or higher.
- `scripts/verify_pipeline.py` (`logs/verify-pipeline.log`), 5 test windows (the 3 worst and windows 0, 1):
  the vision config, the `pixel_values` cast and the processor size of the d3-flash runtime give bitwise equal features.
  The d3-flash runtime replaces the patch-embedding Conv3d with `F.linear` (`d3_runtime.linearize_patch_embed`).
  With that change, the features equal vit-1 exactly (max abs difference 0).
- So the difference is the bfloat16 summation order of the patch embedding, not a pipeline error.
  `encoder-extract` keeps the transformers Conv3d.

## Results

Test frame AUC (117 videos), 95 % video bootstrap (1,000 resamples, seed 20261022). Selected = the state chosen on
validation by each method (decision 7). Diff = paired difference against `qwen35-vit` `patch_mean` at both selected states.

| Encoder | Kind | MIL selected (state) | MIL diff vs baseline | Holmes selected (state) | Holmes diff | kNN selected (state) | kNN diff |
|---|---|---|---|---|---|---|---|
| qwen35-vit | patch_mean | 0.807 [0.756, 0.858] (18) | — | 0.790 [0.725, 0.851] (4) | — | 0.663 [0.594, 0.730] (11) | — |
| qwen35-vit | patch_max | 0.816 [0.767, 0.868] (26) | +0.009 [-0.017, +0.035] | 0.737 [0.671, 0.802] (21) | -0.053 [-0.113, +0.004] | 0.628 [0.547, 0.701] (8) | -0.035 [-0.109, +0.034] |
| internvit-300m | patch_mean | 0.793 [0.739, 0.850] (14) | -0.014 [-0.038, +0.008] | 0.795 [0.745, 0.847] (11) | +0.005 [-0.050, +0.064] | 0.643 [0.559, 0.721] (11) | -0.020 [-0.060, +0.023] |
| internvit-300m | patch_max | 0.797 [0.742, 0.852] (13) | -0.011 [-0.037, +0.016] | 0.771 [0.719, 0.828] (13) | -0.018 [-0.095, +0.071] | 0.585 [0.515, 0.651] (9) | -0.079 [-0.132, -0.019] |
| internvit-300m | cls_mean | 0.804 [0.752, 0.856] (23) | -0.003 [-0.023, +0.016] | 0.819 [0.759, 0.871] (19) | +0.029 [-0.028, +0.086] | 0.667 [0.586, 0.744] (9) | +0.004 [-0.049, +0.059] |
| internvit-300m-head | patch_mean | 0.809 [0.752, 0.861] (0) | +0.001 [-0.026, +0.030] | 0.809 [0.749, 0.862] (0) | +0.019 [-0.017, +0.056] | 0.565 [0.459, 0.656] (0) | -0.098 [-0.185, -0.026] |
| internvit-300m-head | patch_max | 0.800 [0.743, 0.856] (0) | -0.007 [-0.046, +0.027] | 0.762 [0.698, 0.826] (0) | -0.028 [-0.089, +0.027] | 0.604 [0.520, 0.676] (0) | -0.059 [-0.155, +0.026] |
| siglip2-so400m | patch_mean | 0.818 [0.765, 0.871] (21) | +0.011 [-0.019, +0.041] | 0.797 [0.732, 0.854] (14) | +0.007 [-0.033, +0.047] | 0.677 [0.595, 0.751] (9) | +0.014 [-0.026, +0.057] |
| siglip2-so400m | patch_max | 0.827 [0.774, 0.875] (24) | +0.020 [-0.011, +0.054] | 0.746 [0.691, 0.805] (15) | -0.044 [-0.113, +0.031] | 0.663 [0.589, 0.736] (9) | -0.000 [-0.065, +0.075] |
| siglip2-so400m-head | pooler_mean | 0.820 [0.767, 0.870] (0) | +0.013 [-0.022, +0.046] | 0.800 [0.738, 0.858] (0) | +0.010 [-0.047, +0.073] | 0.645 [0.577, 0.716] (0) | -0.018 [-0.095, +0.057] |
| dinov3-vitl | patch_mean | 0.798 [0.739, 0.854] (17) | -0.009 [-0.043, +0.028] | 0.771 [0.707, 0.837] (6) | -0.019 [-0.048, +0.013] | 0.671 [0.571, 0.756] (18) | +0.008 [-0.054, +0.062] |
| dinov3-vitl | patch_max | 0.768 [0.708, 0.825] (14) | -0.040 [-0.079, -0.007] | 0.753 [0.699, 0.809] (23) | -0.037 [-0.103, +0.040] | 0.622 [0.549, 0.696] (11) | -0.041 [-0.117, +0.039] |
| dinov3-vitl | cls_mean | 0.782 [0.722, 0.842] (23) | -0.025 [-0.071, +0.012] | 0.750 [0.689, 0.812] (23) | -0.040 [-0.105, +0.025] | 0.690 [0.612, 0.761] (12) | +0.027 [-0.038, +0.088] |
| vjepa2-vitl | patch_mean | 0.805 [0.748, 0.860] (15) | -0.002 [-0.042, +0.035] | 0.813 [0.748, 0.877] (18) | +0.023 [-0.051, +0.111] | 0.701 [0.615, 0.781] (21) | +0.038 [-0.036, +0.103] |
| vjepa2-vitl | patch_max | 0.759 [0.700, 0.816] (13) | -0.048 [-0.091, -0.011] | 0.783 [0.727, 0.836] (7) | -0.007 [-0.064, +0.050] | 0.674 [0.581, 0.759] (17) | +0.011 [-0.079, +0.092] |

Main findings:

- No encoder beats the baseline (decision 8): no MIL or Holmes interval lies above 0.
  The largest MIL difference is `siglip2-so400m` `patch_max`, +0.020 [-0.011, +0.054].
- At the validation-selected state, all encoders reach MIL frame AUC 0.76 to 0.83; most intervals overlap.
- Intervals below 0: MIL `patch_max` of `dinov3-vitl` (-0.040) and `vjepa2-vitl` (-0.048);
  kNN `internvit-300m` `patch_max` (-0.079) and `internvit-300m-head` `patch_mean` (-0.098).
- `qwen35-vit` `patch_mean` at the last state gives MIL 0.825, the vit-1 value. The validation choice (state 18) gives 0.807.

## Analysis after the run

The in-video localization analyses that started from this run (per category, in video, per video, and everything
after) moved to `../../in_video_localization/` on 2026-10-31. Its `README.md` is the index.

## Notes

- State 0 `cls_mean` of `dinov3-vitl` and `internvit-300m` is the same for every window (the learned CLS embedding,
  std 0), so its AUC is 0.5. This causes the 4 sklearn PCA `invalid value encountered in divide` warnings in
  `logs/fit-dinov3-vitl.log` and `logs/fit-internvit-300m.log`.
- Absolute paths in logs and `fit.json` are replaced with `$S1_FEATURES`, `$S1_ASSETS` and `<repo>`.

## Status

Done.
