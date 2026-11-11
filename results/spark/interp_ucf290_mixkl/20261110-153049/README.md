# Interpolation method at 2x, mix + KL, UCF-Crime test 290 (queued 2026-11-10)

Eric, 2026-11-10: compare interpolation methods at 2x on mix + KL. `run.sh` scores nearest2, bilinear2 and lanczos2
(scripts/sr-score.py --upscale, cv2 INTER_NEAREST / INTER_LINEAR / INTER_LANCZOS4, 640x480); bicubic 2x comes from
`results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c/B`. Each method: memory check, 5-window smoke test,
real run (timeout 6 h). Evaluation: vs mix + KL 1x, vs bicubic 2x, macro AUC.

`queue.sh <WiSE PID>` waits for the WiSE run, runs `results/spark/sr_ucf290_d3nano_ft/20261110-153049/run.sh`, then
this `run.sh`. Logs: `queue.log`, `run.log`.
- interim-2of3/: evaluation of nearest2 and bilinear2 before lanczos2 finished (2026-11-11).
