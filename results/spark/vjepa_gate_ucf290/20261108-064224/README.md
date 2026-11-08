# vjepa_gate_ucf290: V-JEPA 2 linear-head gate for the 2x bicubic up-scale, 290 videos (2026-11-08)

Question: does the calibrated V-JEPA 2 head gate of `vjepa_gate_calib_ucf117` hold on all 290 UCF test videos?
Offline, CPU only. Same script as `qwenvit_gate_ucf290`; only the gate scores differ.

- Gate score: V-JEPA 2 ViT-L holmes head, patch_mean, state 18, lam 0.01, refit on the training windows of
  `encoder_probe_ucf_short_v1_f8` (the original head weights were not saved).
  Scores: `experiment/encoder-probing`, `results/spark/encoder_probe_ucf_test_rest_v1_f8/20261108-063724/refit-290/`.
  Check of the refit on the 117 short videos against the stored scores: max abs diff 0.000758, Spearman 1.000000.
  The features of the other 173 videos (16,706 windows) come from the same encoder, frames and forward path.
- Main setting, fixed before the run: z-score over all previous windows of the video, top 20%. Other rows are references.
- Models: zero-shot d3-nano and `floor_d3nano_c` (mix + KL). Metrics: `scripts/layout-eval.py` functions.

Result, mix + KL, AP (small / medium / large / all): original 22.5 / 37.2 / 29.0 / 47.3;
zscore_all top 20% 24.4 / 39.6 / 27.8 / 48.6 (dAP all [+0.1, +2.9]); upscaled all 26.1 / 37.9 / 25.9 / 48.7.
Full table: `run.log`.

Note: `run.log` of the extraction run in encoder-probing stayed empty until the end, because `grep -v` buffers
its output before `tee`.

Files: `gate290.py`, `config.yaml`, `run.sh`, `run.log`, `eval-zeroshot.json`, `eval-mixkl.json`, `rates.json`.
