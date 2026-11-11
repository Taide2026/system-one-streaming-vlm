# vjepa_gate_online_ucf290: V-JEPA 2 gate with an online threshold, 290 videos (2026-11-08)

Question: the top-20% threshold of vjepa_gate_ucf290 needs all test windows, so it is not online. With a fixed z-score
cutoff c read from the training videos only, does the gate result hold?
Offline, CPU only. Gate scores, causal z-score and metrics as `vjepa_gate_ucf290/20261108-064224`.

- Main, fixed before the run: c = 1.095, the cutoff that up-scales 20% of the windows of the 390 training videos,
  from 5-fold cross-fitted head scores (`train_scores.py` in encoder-probing, run 20261108-063724/train-scores/).
- References: in-sample training scores (c = 1.133); fixed c = 0.5, 1.0, 1.5.
- Training videos are at most 128 s; test videos go up to 4000 s.

Result, mix + KL, AP (small / medium / large / all): original 22.5 / 37.2 / 29.0 / 47.3;
main 24.2 [+0.2, +3.1] / 39.2 / 27.7 / 48.5 [-0.0, +2.6]; 17.9% of the test windows up-scaled. Full table: `run.log`.

Files: `online.py`, `config.yaml`, `run.sh`, `run.log`, `online.json`.
