#!/usr/bin/env bash
# floor2x_d3nano_c (Colab A100, 2026-11-10, T = 0.92) on UCF-Crime test 290: draft = 1x, B = all 8 frames bicubic 2x.
# Eric, 2026-11-10: test this model first after the WiSE run. Then evaluate against mix + KL.
set -euo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov
R=results/spark/sr_ucf290_d3nano_ft/20261110-153049
AD=results/colab/floor2x_d3nano_c/20261110-a100
F=$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
M=results/spark/frames_ucf290_f8/20261105-180529/manifest.json
E=results/spark/tgm_size_ucf290/20261105-200402/jev-hard-events.jsonl
MIX=results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c
SCORE=(uv run --frozen python scripts/sr-score.py --adapter $AD --frames-dir $F --index $I)

avail=$(free -g | awk '/^Mem:/ {print $7}'); echo "available ${avail} GB"
[ "$avail" -ge 15 ] || { echo "not enough memory, stop"; exit 1; }
S=results/smoke/floor2x_ft/$(date -u +%Y%m%d-%H%M%S)
timeout 15m "${SCORE[@]}" --upscale bicubic2 --limit-windows 5 --out $S/B/scores.jsonl
echo "smoke ok $(date -u +%FT%TZ)"

echo "draft start $(date -u +%FT%TZ)"
timeout 3h "${SCORE[@]}" --upscale none --out $R/floor2x_d3nano_c/draft/scores.jsonl
echo "B start $(date -u +%FT%TZ)"
timeout 6h "${SCORE[@]}" --upscale bicubic2 --out $R/floor2x_d3nano_c/B/scores.jsonl
echo "scoring done $(date -u +%FT%TZ)"

D=$R/floor2x_d3nano_c
uv run --frozen python scripts/layout-eval.py --manifest $M --events $E --draft $MIX/draft/scores.jsonl \
  --run mix2x:$MIX/B/scores.jsonl f2x1x:$D/draft/scores.jsonl f2x2x:$D/B/scores.jsonl --out $R/eval-vs-mixkl.json 2>&1 | tee $R/eval-vs-mixkl.log
uv run --frozen python scripts/layout-eval.py --manifest $M --events $E --draft $D/draft/scores.jsonl \
  --run f2x2x:$D/B/scores.jsonl --out $R/eval-2x-vs-1x.json 2>&1 | tee $R/eval-2x-vs-1x.log
uv run --frozen python results/spark/sr_ucf290_d3nano_zs/20261106-040502/scripts/score-shift.py \
  --draft $D/draft/scores.jsonl --cond $D/B/scores.jsonl --manifest $M --events $E 2>&1 | tee $R/score-shift-B.log
uv run --frozen python results/spark/macro_auc_ucf290/20261110-133458/macro_auc.py $M $E $R/macro.json \
  mix1x=$MIX/draft/scores.jsonl mix2x=$MIX/B/scores.jsonl f2x1x=$D/draft/scores.jsonl f2x2x=$D/B/scores.jsonl 2>&1 | tee $R/macro.log
echo "all done $(date -u +%FT%TZ)"
