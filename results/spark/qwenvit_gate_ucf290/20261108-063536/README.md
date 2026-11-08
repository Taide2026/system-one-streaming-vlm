# qwenvit_gate_ucf290: Qwen3.5 ViT linear-head gate for the 2x bicubic up-scale, 290 videos (2026-11-08)

Question: does the calibrated head gate of `vjepa_gate_calib_ucf117` hold on all 290 UCF test videos, with the stored
290-video scores of a Qwen3.5 ViT head (no V-JEPA 2 features exist for the 173 other videos yet)?
Offline, CPU only.

- Gate score: "08 qwen35-vit-video bce" from the encoder-probing archive
  (`in_video_localization/21-test-290/scores.npz`; plain BCE head, layer and lam chosen on training validation videos).
  Rows mapped by (video, start_s): 18,390 of 18,390 windows matched, labels equal.
- Main setting, fixed before the run: z-score over all previous windows of the video, top 20%. Other rows are references.
- Models: zero-shot d3-nano and `floor_d3nano_c` (mix + KL). Metrics: `scripts/layout-eval.py` functions.

Files: `gate290.py`, `config.yaml`, `run.sh`, `run.log`, `eval-zeroshot.json`, `eval-mixkl.json`, `rates.json`.
