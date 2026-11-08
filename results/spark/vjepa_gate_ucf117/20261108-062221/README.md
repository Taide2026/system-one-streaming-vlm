# vjepa_gate_ucf117: V-JEPA 2 linear-head gate for the 2x bicubic up-scale (2026-11-08)

Question: does a cheap per-window gate from a different encoder up-scale fewer negative windows than the
previous-window gate (`results/spark/prevgate_ucf290/20261108-060410/`)?
Offline, CPU only. It reuses saved per-window scores: no model runs.

- Gate: V-JEPA 2 ViT-L holmes head, patch_mean, layer 18 (chosen on validation, same rule as `encoder-compare`).
  Source: encoder-probing run `encoder_probe_ucf_short_v1_f8/run-1/fit-vjepa2-vitl` (trained on UCF train videos).
- Videos: its 117 short test videos (<= 49.9 s), 1,684 windows, 52 of the 156 events. Tier bounds from all 156 events.
- Rule: window t uses the VLM's 2x score when its own gate score is in the top 10/20/30/50% of the 1,684 windows.
  The thresholds are read on the test set.
- Models: zero-shot d3-nano and `floor_d3nano_c` (mix + KL). Metrics: `scripts/layout-eval.py` functions.

Files: `cascade.py`, `config.yaml`, `run.sh`, `run.log`, `eval-zeroshot.json`, `eval-mixkl.json`, `rates.json`.
