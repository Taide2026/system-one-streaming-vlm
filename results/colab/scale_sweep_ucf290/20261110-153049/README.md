# Scale sweep on Colab A100, UCF-Crime test 290 (prepared 2026-11-10)

Eric, 2026-11-10: up-scale factors 1.25 / 1.5 / 1.75 / 3, bicubic, all 8 full frames, on mix + KL
(floor_d3nano_c) and fov2 (fov2_d3nano_c); plus fov2 at 1x (no per-window fov2 1x scores existed). 9 jobs.
1x and 2x of mix + KL and 2x of fov2: results/spark/sr_ucf290_d3nano_ft/20261106-142617/.

Drive folder s1-colab-scale-score (id 1-hm9GQgErrT_ZfS7w4F2Byyb_0agu4EG): code.tar.gz, adapters.tar, setup.sh,
score.sh, run-all.sh (md5 checked after upload). Frames are read from s1-colab-wise-score.
Output: s1-colab-scale-score/out/<model>/<upscale>/; partial copies every 10 min in out-partial/.
