# nano4 runs of 2026-11-15: interpolated against real pixels, more models

Whole frame, Macro-AUC (mean per-video AUC over videos with both labels). Differences: mean ± half the width of the
95% bootstrap interval over videos. Base 180p; "bicubic 2x" is the 180p frame enlarged; "real 360p" is the camera frame
area-downscaled to 360p. MSAD per-window scores are not in the repository (research-only data); its run summaries are.

## ComplexVAD, 113 test videos (110 with both labels), 26,810 windows

| model | run names | 180p | bicubic 2x | real 360p | real − bicubic |
|---|---|---|---|---|---|
| d3-nano, zero-shot | `cvadall_zs_*` | 76.4 | 79.8 (+3.4 ± 1.7) | 82.0 (+5.6 ± 2.1) | +2.2 ± 1.5 |
| d3-flash, zero-shot | `cvadall_flzs_*` | 71.6 | 71.3 (−0.3 ± 2.4) | 70.3 (−1.3 ± 3.4) | −1.0 ± 2.1 |
| d3-flash, UCF fine-tune, seed 1 | `cvadall_flf1_*` | 82.1 | 84.9 (+2.8 ± 2.0) | 82.8 (+0.7 ± 2.3) | −2.1 ± 1.6 |
| d3-flash, UCF fine-tune, seed 2 | `cvadall_flf2_*` | 81.2 | 83.0 (+1.7 ± 1.5) | 82.2 (+1.0 ± 2.5) | −0.8 ± 1.8 |
| Qwen3-VL-4B, neutral question | `svcvad_qwen3-vl-4b_yn_*` | 69.5 | 69.9 (+0.4 ± 1.6) | 72.3 (+2.9 ± 2.6) | +2.4 ± 2.1 |
| Qwen3-VL-4B, crime-list question | `svcvad_qwen3-vl-4b_*` | 39.1 | 36.2 | 46.3 | below chance |

Fine-tuning d3-flash on UCF-Crime raises its 180p score by +10.5 ± 3.2 and +9.6 ± 3.9 over zero-shot.
Qwen3-VL-4B: neutral question against crime-list question at 180p, +30.3 ± 5.4 (better in 91 of 110 videos).
Gains by event size (videos split into thirds by tokens on the event): `tokthird-cvad.txt`, script `scripts/tokthird.py`.

## MSAD, 221 test videos (76 with both labels), 2,121 windows

| model | run names | 180p | bicubic 2x | real 360p | real − bicubic |
|---|---|---|---|---|---|
| d3-nano, zero-shot | `msad_zs_*` | 78.6 | 80.5 (+2.0 ± 4.6) | 83.7 (+5.2 ± 4.2) | |
| d3-flash, zero-shot | `msad_flzs_*` | 84.6 | 86.5 (+1.9 ± 4.9) | 85.4 (+0.8 ± 5.9) | −1.1 ± 3.8 |
| d3-flash, UCF fine-tune, seed 1 | `msad_flf1_*` | 81.8 | 85.9 (+4.1 ± 5.0) | 84.5 (+2.7 ± 5.7) | −1.4 ± 5.0 |
| d3-flash, UCF fine-tune, seed 2 | `msad_flf2_*` | 84.3 | 87.7 (+3.4 ± 4.1) | 84.0 (−0.3 ± 3.6) | −3.7 ± 2.8 |
| Qwen3-VL-4B, neutral question | `svmsad_qwen3-vl-4b_yn_*` | 82.5 | 83.9 (+1.4 ± 4.1) | 82.4 (−0.1 ± 4.1) | −1.5 ± 3.8 |

## Also added (from 2026-11-13/14)
`gran-1113.json` (small in space, partial in time), `stream-1113.json` (stream metrics), `latsh/771911.json` (latency),
`scripts/tokslope.py`, `scripts/split.py`, `scripts/oracle-recalc.py`, `scripts/msad-train-prep.py`.
Job scripts: `jobs/nano4-upx/{second,second2,second3,qwenyn,msad2,flft1,flft2,flft1g,msadflft,svcvad}.sbatch`.
