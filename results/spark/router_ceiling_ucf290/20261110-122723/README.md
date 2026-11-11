# router_ceiling_ucf290 / 20261110-122723

Question (Eric, 2026-11-10): how accurate must a 1x/2x router be for mix + KL to reach AP all 50?
CPU only, GB10. A simulated router up-scales each event window with probability TPR and each negative with
probability FPR, at random (independent of the scores). AP all = mean of 20 draws. Test 290 videos, 18,390 windows.

| File | Content |
|---|---|
| ceiling.py | Simulation |
| ceiling.log, ceiling.json | AP grid (TPR x FPR in steps of 0.1) |

    uv run --frozen python ceiling.py <mix 1x scores> <mix 2x scores> ceiling.json

Inputs: results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c/{draft,B}/scores.jsonl.
Check row reproduces 1x 47.3, all 2x 48.7, oracle 56.7.
