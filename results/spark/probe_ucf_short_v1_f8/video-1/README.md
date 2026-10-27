# probe_ucf_short_v1_f8 / video-1

vit-1 with video input instead of image input (2026-10-27). Proposal: `.specs/changes/2026-10-27-vit-video-input/proposal.md` (local).

Config: `config.yaml` in this folder (the base config plus the settings of this run).

## Input

- Image input (vit-1): each frame is its own image, grid 8 x [1, 16, 20], 2,560 patches. The Conv3d sees the same frame twice.
- Video input (this run): `video_processor` + `get_video_features`, grid [[4, 16, 20]], 1,280 patches.
  The Conv3d (kernel 2 x 16 x 16) sees adjacent frame pairs (f0, f1), (f2, f3), ... so it can see motion.
- Same windows, 8 frames, ViT weights, taps, poolings, probes and bootstrap as vit-1.

## Steps

| Step | Command | Result |
|---|---|---|
| Check | `scripts/extract_vit_video.py --run check --check 3` | doubled-frame 16-frame video = image path, max abs diff 0 (`logs/check.log`) |
| Smoke | `--limit-windows 5` | peak GPU 15.7 GiB (`logs/smoke.log`) |
| Extract | `scripts/extract_vit_video.py --run video-1` (timeout 9000 s) | 3,937 s for 12,772 windows, peak GPU 15.7 GiB, RSS 17 GB |
| Fit | `scripts/fit_gpu.py` (vit-1) on VM, then VV, cap 8 GiB | 86 s and 1,577 s, peak GPU 0.23 / 0.11 GiB |
| Compare | `scripts/compare_video.py` | `compare_video.json`, `video_<method>.png`, `logs/compare.log` |

Features: `$S1_FEATURES/probe_ucf_short_v1_f8/{VV,VM}/video-1` (float32; visual = token mean, last = token max).

## Status

Done 2026-10-27 (extract 04:27 to 05:32 UTC, fits to about 06:00 UTC).

## Results (test frame AUC, 117 videos, 95 % paired video-bootstrap interval, peak on test)

| Probe, pooling | image (vit-1) peak | video peak | peak diff (video - image) | state 27 diff |
|---|---|---|---|---|
| MIL, mean | 0.825 @27 [0.776, 0.875] | 0.820 @26 [0.772, 0.871] | -0.005 [-0.017, +0.008] | -0.011 [-0.024, +0.000] |
| MIL, max | 0.817 @19 | 0.819 @17 | +0.002 [-0.017, +0.022] | |
| Holmes, mean | 0.819 @27 [0.767, 0.871] | 0.820 @27 [0.767, 0.871] | +0.001 [-0.012, +0.015] | +0.001 [-0.012, +0.015] |
| Holmes, max | 0.794 @26 | 0.813 @19 | +0.018 [-0.032, +0.068] | |
| kNN, mean | 0.666 @12 | 0.652 @10 | -0.014 [-0.042, +0.012] | +0.017 [-0.018, +0.055] |
| merger MIL, mean | 0.819 | 0.805 | -0.014 [-0.029, -0.000] | |
| merger Holmes, mean | 0.744 | 0.750 | +0.006 [-0.012, +0.024] | |

Video ViT peak - P(Yes) of C: MIL mean -0.040 [-0.077, +0.003], Holmes mean -0.040 [-0.086, +0.005].

Reading:
- Video input does not raise the ViT probe. All peak differences are within about ±0.02 and their intervals include 0.
- The merger with video input is slightly lower for MIL mean (-0.014, interval touches 0).
- In the MIL curve, video input is about 0.01 to 0.02 lower in the early and middle blocks and equal from block 18 on.
- The video path has half the tokens (1,280 vs 2,560). It gives the same information at half the LM input cost.
- The ceiling of about 0.82 to 0.83 at the ViT output does not come from the missing motion in the patch embedding.
- Limit: only the Conv3d sees motion, inside each frame pair (frames 0.25 s apart: 8 frames in a 2 s window). ViT attention stays inside each temporal slice, so pairs do not see each other in the ViT.
