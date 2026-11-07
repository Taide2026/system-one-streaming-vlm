# sr_ucf290_d3nano_ft / 20261106-142617

Full-frame up-scale test with two fine-tuned d3-nano adapters from `bear7011/streaming-vad-decision-adapters`
(revision df17125220449daa89a491caf2299e3d1f84a1f7), downloaded to `results/pro6000/` (gitignored weights).

| Model | Run | Training input |
|---|---|---|
| `floor_d3nano_c` | 20261031-045123 | multi-task mix + KL 0.5, 8 full frames |
| `fov2_d3nano_c` | 20261101-183539 | same recipe, foveated input (4 full + 4 TGM 1/3 crops, gate 0.045) |

Per model: `B/` (bicubic 2x, 640x480), `C/` (Real-ESRGAN x2plus frames from
`$S1_FEATURES/frames/frames_ucf290_f8_sr2/20261106-040502/`). fov2 is scored on 8 full frames here, not its training layout.
Zero-shot counterpart: `results/spark/sr_ucf290_d3nano_zs/20261106-040502/`. Smoke test: `results/smoke/full_frame_sr_ft/20261106-142246/`.
Draft (original 320x240 frames) was dropped (Eric, 2026-11-06): the classmate's jev-hard tier table gives window-level numbers for both
models. `floor_d3nano_c/draft/` was later completed by `run-draft.sh` (Eric, 2026-11-07), resuming the 3,080 windows of the stopped run.

## Results (2026-11-07)

Window AUC / AP (%), same tiers and bootstrap as the zero-shot run. floor_d3nano_c draft on the GB10 reproduces the
classmate's pro6000 tier table within 0.2 (84.0 / 22.7, 90.9 / 37.3, 92.2 / 29.0, 87.9 / 47.4).

floor_d3nano_c [ΔAP vs its own draft]:

| Input | small | medium | large | all |
|---|---|---|---|---|
| draft (original) | 83.9 / 22.5 | 90.8 / 37.2 | 92.3 / 29.0 | 87.9 / 47.3 |
| B bicubic 2x | 85.4 / 26.1 [+0.7, +5.9] | 92.1 / 37.9 [-2.3, +4.8] | 92.3 / 25.9 [-6.3, +0.7] | 89.0 / 48.7 [-0.8, +3.6] |
| B + gate 0.045 | 85.7 / 27.0 [+1.6, +6.7] | 91.5 / 36.7 [-3.1, +2.5] | 91.4 / 24.3 [-7.8, -0.8] | 88.8 / 48.3 [-1.0, +3.1] |
| C Real-ESRGAN x2 | 84.4 / 23.3 [-2.4, +4.2] | 92.0 / 35.6 [-6.5, +4.1] | 91.9 / 23.2 [-9.5, -0.9] | 88.4 / 46.8 [-3.6, +2.5] |
| C + gate 0.045 | 84.4 / 24.8 [-1.2, +5.8] | 91.6 / 34.4 [-7.1, +1.8] | 91.8 / 21.4 [-12.3, -1.3] | 88.3 / 46.6 [-3.6, +2.2] |

fov2_d3nano_c (no per-window draft; the classmate's table with its own foveated input: 85.7 / 27.1, 91.4 / 36.4, 91.4 / 24.0, 88.9 / 47.8):

| Input | small | medium | large | all |
|---|---|---|---|---|
| B bicubic 2x | 86.3 / 28.2 | 92.5 / 38.9 | 91.8 / 22.5 | 89.4 / 49.2 |
| C Real-ESRGAN x2 | 85.9 / 26.8 | 92.3 / 37.7 | 91.5 / 21.1 | 89.1 / 48.0 |

C vs B ΔAP: floor small [-5.6, +0.4], medium [-5.5, +1.7], large [-5.0, +1.1], all [-4.0, +0.5];
fov2 small [-3.9, +1.3], medium [-4.7, +3.3], large [-4.7, +2.7], all [-3.6, +1.4].
Forward on the GB10: draft 0.26, B 0.71, C 0.71-0.72 s/window; peak 4.60 GB.

| Files (per model folder) | Content |
|---|---|
| `eval-draft.log` (floor) | draft AUC / AP |
| `eval-vs-draft-gate045.*` (floor) | B, C and gated versions vs draft |
| `eval-C-vs-B.*` | C vs B |
| `score-shift-B.log` (floor), `score-shift-C-vs-B.log` | score shifts of normal and event windows |
