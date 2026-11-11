# Scale sweep, GB10 part (2026-11-11)

Eric, 2026-11-11: up-scale factor sweep (bicubic, all 8 full frames) on mix + KL and fov2, 9 jobs. 6 jobs run on
Colab A100 40 GB (results/colab/scale_sweep_ucf290/20261110-153049, run-6.sh: 1.5x, 1.75x, 3x of both models).
This folder: fov2 1x (`none`), mix + KL 1.25x, fov2 1.25x. `run.sh <PID>` waits for the interpolation queue
(results/spark/interp_ucf290_mixkl/20261110-153049/queue.sh), then runs each job with a memory check and a 5-window smoke test.
Log: run.log. Output: <model>/<upscale>/scores.jsonl.
