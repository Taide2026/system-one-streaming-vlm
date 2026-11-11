#!/usr/bin/env bash
# Interpolation at 2x on mix + KL (floor_d3nano_c), UCF-Crime test 290 (Eric, 2026-11-10).
# bicubic 2x already exists: results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c/B.
set -euo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov
R=results/spark/interp_ucf290_mixkl/20261110-153049
AD=results/pro6000/floor_d3nano_c/20261031-045123
F=$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
M=results/spark/frames_ucf290_f8/20261105-180529/manifest.json
E=results/spark/tgm_size_ucf290/20261105-200402/jev-hard-events.jsonl
MIX=results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c
SCORE=(uv run --frozen python scripts/sr-score.py --adapter $AD --frames-dir $F --index $I)

for m in nearest2 bilinear2 lanczos2; do
  avail=$(free -g | awk '/^Mem:/ {print $7}'); echo "available ${avail} GB"
  [ "$avail" -ge 15 ] || { echo "not enough memory, stop"; exit 1; }
  S=results/smoke/interp_mixkl/$m-$(date -u +%Y%m%d-%H%M%S)
  timeout 15m "${SCORE[@]}" --upscale $m --limit-windows 5 --out $S/scores.jsonl
  echo "$m smoke ok, start $(date -u +%FT%TZ)"
  timeout 6h "${SCORE[@]}" --upscale $m --out $R/$m/scores.jsonl
  echo "$m done $(date -u +%FT%TZ)"
done

uv run --frozen python scripts/layout-eval.py --manifest $M --events $E --draft $MIX/draft/scores.jsonl \
  --run bicubic2:$MIX/B/scores.jsonl nearest2:$R/nearest2/scores.jsonl bilinear2:$R/bilinear2/scores.jsonl lanczos2:$R/lanczos2/scores.jsonl \
  --out $R/eval-vs-mix1x.json 2>&1 | tee $R/eval-vs-mix1x.log
uv run --frozen python scripts/layout-eval.py --manifest $M --events $E --draft $MIX/B/scores.jsonl \
  --run nearest2:$R/nearest2/scores.jsonl bilinear2:$R/bilinear2/scores.jsonl lanczos2:$R/lanczos2/scores.jsonl \
  --out $R/eval-vs-bicubic2.json 2>&1 | tee $R/eval-vs-bicubic2.log
uv run --frozen python results/spark/macro_auc_ucf290/20261110-133458/macro_auc.py $M $E $R/macro.json \
  mix1x=$MIX/draft/scores.jsonl bicubic2=$MIX/B/scores.jsonl nearest2=$R/nearest2/scores.jsonl \
  bilinear2=$R/bilinear2/scores.jsonl lanczos2=$R/lanczos2/scores.jsonl 2>&1 | tee $R/macro.log
echo "all done $(date -u +%FT%TZ)"
