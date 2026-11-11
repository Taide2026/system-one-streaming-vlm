# vjepa_gate_split_ucf290: V-JEPA 2 gate result split into 117 short and 173 other videos (2026-11-08)

Question: the gate gain (zscore_all, top 20%, mix + KL) is all dAP +3.7 on the 117 short videos but +1.3 on 290.
Is the gain smaller on the other 173 videos, and where does the up-scale budget go there?
Offline, CPU only. Gate, calibration and metrics as `vjepa_gate_ucf290/20261108-064224`.

Threshold rules: global top 20% of all 18,390 windows (as in vjepa_gate_ucf290), and top 20% within each subset.

Files: `split.py`, `config.yaml`, `run.sh`, `run.log`, `split.json`.
