# probe_ucf_short_v1_f8 / frames-1

Config: `config.yaml` in this folder (the base config plus the settings of this run).

Derived experiment of `../vit-1` (2026-10-27): does averaging the 8 frames of a window hide time information?

The vision tower encodes each frame as its own image (image_grid_thw: 8 rows, t = 1), so the ViT sees no motion.
vit-1 averaged all tokens of the 8 frames. Here the tokens are averaged per frame, and the order is kept.

## Features

| Folder | Source | Kind "last" | Kind "visual" |
|---|---|---|---|
| `$S1_FEATURES/probe_ucf_short_v1_f8/FV/frames-1/` | ViT block 27 output | concat of the 8 frame means (9,216 dims) | mean and std over frames (2,304 dims) |
| `$S1_FEATURES/probe_ucf_short_v1_f8/FM/frames-1/` | merger output | concat (32,768 dims) | mean and std (8,192 dims) |

`frames.npy` (n, 8, d) keeps the raw per-frame means. float32. The frame average equals the vit-1 mean (max relative difference 7e-8 on the smoke windows).

## Steps

| Step | Command (worktree root, `uv run python`) | Result |
|---|---|---|
| Smoke | `scripts/extract_frames.py --run smoke --limit-windows 5` | peak GPU 15.8 GiB, RSS 17 GB |
| Extract | `scripts/extract_frames.py --run frames-1` | 3,339 s, 12,772 windows |
| Fit | `../vit-1/scripts/fit_gpu.py <features> fit-FV 8`, then fit-FM | 101 s / 241 s; peak GPU 0.44 / 1.42 GiB; RSS 4 / 10 GB |
| Eval | `scripts/eval_frames.py` (paired with vit-1 frame averages, same 1,000 video resamples) | `eval_frames.json`, `logs/eval.log` |

## Results (test frame AUC; difference to the 8-frame average with 95 % interval)

| Position · probe | 8-frame average | concat | mean + std |
|---|---|---|---|
| ViT 27 · MIL | 0.825 | 0.820, −0.005 [−0.015, +0.003] | 0.826, +0.001 [−0.006, +0.008] |
| merger · MIL | 0.819 | 0.820, +0.001 [−0.004, +0.005] | 0.822, +0.003 [−0.004, +0.009] |
| ViT 27 · Holmes | 0.819 | 0.782, −0.037 [−0.061, −0.017] | 0.814, −0.004 [−0.015, +0.009] |
| merger · Holmes | 0.744 | 0.682, −0.062 [−0.099, −0.028] | 0.729, −0.015 [−0.049, +0.014] |
| ViT 27 · kNN | 0.468 | 0.613, +0.146 [+0.021, +0.268] | 0.545, +0.077 [−0.014, +0.178] |
| merger · kNN | 0.499 | 0.542, +0.043 [+0.009, +0.083] | 0.532, +0.033 [+0.009, +0.060] |

Reading:
- MIL does not change (±0.005). A linear probe on the concat can also read frame-to-frame differences, and they do not help.
- The 8-frame average does not hide the information. The per-frame features carry about the same amount.
- Holmes gets worse on the concat, possibly from overfitting in high dimensions.
- kNN improves but stays far below 0.82.
