# probe_ucf_short_v1_f8 / vit-1

Config: `config.yaml` in this folder (the base config plus the settings of this run).

Spike (2026-10-26): probe every state of the vision tower (ViT), before the LM.
Question: is the anomaly information already limited in the ViT output, or is it lost between the ViT and the LM?

Same config, windows, frames, processor, probes and bootstrap as run-2. Only the feature source changes.

## Features

The vision tower of d3-flash and Qwen3.5-9B has the same weights (run-1 extract: `visual_weights_equal = true` for A and B).
So one extraction covers A, B and C.

| Folder | States | Dims | Content |
|---|---|---|---|
| `$S1_FEATURES/probe_ucf_short_v1_f8/V/vit-1/` | 28 | 1152 | state 0 = patch embedding + position embedding; state s = output of ViT block s − 1 |
| `$S1_FEATURES/probe_ucf_short_v1_f8/M/vit-1/` | 1 | 4096 | merger output (the tokens the LM receives) |

- Each state is pooled over all patch tokens of the 8 frames, in two ways.
- `visual.npy` = token mean (the same pooling as the LM "visual" kind).
- `last.npy` = token **max**. The ViT has no last token. The file name is kept so `probe-fit` reads it unchanged.
- Storage is float32. ViT activations reach about 1.3e4 (max pool), so float16 is not safe.

## Check before the run

`logs/check.log`: on 3 test windows, the vision-only path (`image_processor` + `get_image_features`) gives the same
ViT and merger features as the full runtime forward. The max abs difference is 0.

## Steps

| Step | Command | Log |
|---|---|---|
| Check | `scripts/extract_vit.py --run check --check 3` | `logs/check.log` |
| Smoke | `scripts/extract_vit.py --run smoke --limit-windows 5`: 0.27 s per window, peak GPU 15.8 GiB, RSS 17 GB | `logs/smoke.log` |
| Extract | `scripts/extract_vit.py --run vit-1` (timeout 9000 s) | `logs/extract.log` |
| Fit | `scripts/fit_gpu.py <features> fit-M 8`, then fit-V (= `probe-fit --device cuda`, cap 8 GiB) | `logs/fit-M.log`, `logs/fit-V.log` |
| Compare | `scripts/compare_vit.py` | `logs/compare.log` |

Run every script from the worktree root with `uv run python`.

## Outputs

| Path | Content |
|---|---|
| `fit-V/`, `fit-M/` | `probe-fit` outputs; kind "visual" = mean, "last" = max |
| `compare_vit.json` | frame AUC per state with 95 % interval, test peaks, paired differences against C (run-2) |
| `vit_<method>.png` | frame AUC per ViT state, merger as the last point, C references as lines |

## Status

| Step | Status |
|---|---|
| Extract | done 2026-10-26 14:58 UTC; 3,419 s for 12,772 windows (0.27 s per window); peak GPU 15.8 GiB |
| Fit | started late: the wait loop of the first chain matched its own command line (`pgrep -f`) and never ended. Killed 2026-10-27 02:19 UTC. fit-M 84 s, fit-V 1,590 s, peak GPU 0.23 / 0.11 GiB |
| Compare | done 2026-10-27 |

## Results (test frame AUC, 117 videos, 95 % video-bootstrap interval, peak on test)

| Probe | ViT best (mean) | merger | LM state 0 visual (C) | LM best (C) | P(Yes) C |
|---|---|---|---|---|---|
| MIL | 0.825 @27 [0.777, 0.875] | 0.819 | 0.819 | 0.856 visual @18 | 0.860 |
| Holmes | 0.819 @27 [0.767, 0.871] | 0.744 | 0.738 | 0.864 last @23 | |
| kNN | 0.666 @12 (max: 0.676 @5) | 0.499 | 0.500 | 0.846 last @21 | |

Paired differences (MIL, mean pooling):
- ViT 27 − LM state 0 visual: +0.006 [−0.009, +0.021].
- merger − LM state 0 visual: +0.000 [−0.003, +0.004].
- ViT 27 − LM best visual (@18): −0.031 [−0.061, −0.001].
- ViT 27 − P(Yes) C: −0.035 [−0.076, +0.009].

Reading:
- The MIL information that reaches the LM is set at the last ViT block. The merger and LM layer 1 do not lose it.
- The LM adds about 0.03 (borderline significant).
- Holmes and kNN drop at the merger (Holmes 0.819 → 0.744, kNN 0.67 → 0.50) and recover in later LM layers.
- Max pooling over tokens is not better than mean pooling.
- The ViT encodes each frame as its own image (image_grid_thw t = 1 per frame), so it sees no motion.
- Caveat: the Holmes ViT curve is noisy; its state-27 peak (0.819) is above state 26 (0.772).

Follow-ups: `../mlp-1/` (MLP probe), `../frames-1/` (per-frame read-out), `../video-1/` (video input: adjacent-frame pairs in the Conv3d; no gain).
