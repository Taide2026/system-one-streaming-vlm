# vjepa_gate_calib_ucf117: causal per-video calibration of the V-JEPA 2 gate (2026-11-08)

Question: does a streaming-safe per-video calibration of the V-JEPA 2 holmes score make it a better up-scale gate?
Offline, CPU only. Same 117 videos, 1,684 windows, head and VLM scores as `results/spark/vjepa_gate_ucf117/20261108-062221/`.

- Calibrations (past windows of the same video only): minus the mean of the previous 5 / 10 / all windows;
  z-score over the previous 10 / all windows. A window without enough past is never up-scaled.
- Gate: top 20% / 30% of the calibrated score. Thresholds and the variants are read on the test set:
  10 gate variants were compared, so the best one is optimistic.

Files: `calibrate.py`, `config.yaml`, `run.sh`, `run.log`, `eval-zeroshot.json`, `eval-mixkl.json`, `rates.json`.
