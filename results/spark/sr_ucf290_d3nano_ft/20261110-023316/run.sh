#!/usr/bin/env bash
# floor2x_hn_d3nano_c on test 290 (Eric, 2026-11-10): "跑完直接幫我測效果 1x 2x都測".
# Recipe: mix + KL plus 50% anomaly-video negatives, 30% bicubic 2x in each of 3 groups (training run
# results/spark/floor2x_hn_d3nano_c/20261109-134518). draft = 1x, B = all 8 frames bicubic 2x.
# Success criteria: anomaly-video negative mean shift 1x -> 2x below mix + KL's +0.056; large AP drop at 2x below -3.1.
set -euo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov
R=results/spark/sr_ucf290_d3nano_ft/20261110-023316
A=results/spark/floor2x_hn_d3nano_c/20261109-134518
F=$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
M=results/spark/frames_ucf290_f8/20261105-180529/manifest.json
E=results/spark/tgm_size_ucf290/20261105-200402/jev-hard-events.jsonl
MIX=results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c
TRAIN_PID=1469946

# 1. wait for the training (cap 3 h)
for _ in $(seq 1 1080); do kill -0 $TRAIN_PID 2>/dev/null || break; sleep 10; done
if kill -0 $TRAIN_PID 2>/dev/null; then echo "training still running after 3 h, stop"; exit 1; fi
tail -3 $A.log
for f in adapter readout.safetensors calibration.json; do
  [ -e $A/$f ] || { echo "missing $A/$f, stop"; exit 1; }
done

# 2. memory check: expected peak about 5 GB (fovfull2x run: 4.60 GB), need 15 GB available
avail=$(free -g | awk '/^Mem:/ {print $7}')
echo "available ${avail} GB"
[ "$avail" -ge 15 ] || { echo "not enough memory, stop"; exit 1; }

# 3. smoke test, 5 windows each
S=results/smoke/floor2x_hn_ft/$(date -u +%Y%m%d-%H%M%S)
timeout 15m uv run --frozen python scripts/sr-score.py --frames-dir $F --index $I --upscale none --adapter $A --limit-windows 5 --out $S/draft/scores.jsonl
timeout 15m uv run --frozen python scripts/sr-score.py --frames-dir $F --index $I --upscale bicubic2 --adapter $A --limit-windows 5 --out $S/B/scores.jsonl
echo "smoke ok $(date -u +%FT%TZ)"

# 4. real runs
echo "draft start $(date -u +%FT%TZ)"
timeout 3h uv run --frozen python scripts/sr-score.py --frames-dir $F --index $I --upscale none --adapter $A --out $R/draft/scores.jsonl
echo "B start $(date -u +%FT%TZ)"
timeout 6h uv run --frozen python scripts/sr-score.py --frames-dir $F --index $I --upscale bicubic2 --adapter $A --out $R/B/scores.jsonl
echo "scoring done $(date -u +%FT%TZ)"

# 5. evaluation (CPU)
uv run --frozen python scripts/layout-eval.py --manifest $M --events $E --draft $MIX/draft/scores.jsonl \
  --run mix2x:$MIX/B/scores.jsonl hn1x:$R/draft/scores.jsonl hn2x:$R/B/scores.jsonl --out $R/eval-vs-mixkl.json 2>&1 | tee $R/eval-vs-mixkl.log
uv run --frozen python scripts/layout-eval.py --manifest $M --events $E --draft $R/draft/scores.jsonl \
  --run hn2x:$R/B/scores.jsonl --out $R/eval-2x-vs-1x.json 2>&1 | tee $R/eval-2x-vs-1x.log
uv run --frozen python results/spark/sr_ucf290_d3nano_zs/20261106-040502/scripts/score-shift.py \
  --draft $R/draft/scores.jsonl --cond $R/B/scores.jsonl --manifest $M --events $E 2>&1 | tee $R/score-shift-B.log
uv run --frozen python results/spark/frame_auc_ucf290/20261109-153701/frame-auc.py $M $R/frame-auc.json \
  mix1x=$MIX/draft/scores.jsonl mix2x=$MIX/B/scores.jsonl hn1x=$R/draft/scores.jsonl hn2x=$R/B/scores.jsonl 2>&1 | tee $R/frame-auc.log
echo "all done $(date -u +%FT%TZ)"
