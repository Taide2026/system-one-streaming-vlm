# floor2x_d3nano_c on UCF-Crime test 290, 1x and 2x (queued 2026-11-10)

Adapter: `results/colab/floor2x_d3nano_c/20261110-a100` (Colab A100, 626 steps, T = 0.92, finished 2026-11-10T15:09Z).
Training config: `train-config.yaml` (copy of configs/floor2x_d3nano_c.yaml). `run.sh` scores draft (1x) and
B (bicubic 2x), then evaluates against mix + KL. Started by `results/spark/interp_ucf290_mixkl/20261110-153049/queue.sh`
after the WiSE run. Log: `run.log`.

2026-11-11: done 2026-11-10T22:05Z. draft 4605 s, B 13045 s, peak 4.60 GB. Results: eval-vs-mixkl.log, eval-2x-vs-1x.log,
eval-vs-mix2x.log (f2x 2x vs mix + KL 2x), score-shift-B.log, macro.log.
