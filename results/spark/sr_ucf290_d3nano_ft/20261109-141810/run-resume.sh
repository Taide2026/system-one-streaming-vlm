#!/usr/bin/env bash
# Resume the WiSE-FT alpha 0.5 scoring on test 290 (Eric, 2026-11-10: "繼續在本地測...WiSE alpha=0.5").
# draft = original frames, resumed from the 1,578 windows of 2026-11-09; B = all 8 frames bicubic 2x.
# Then evaluate: wise 1x and 2x against mix + KL 1x, wise 2x against wise 1x, score shift, frame AUC / AP.
set -euo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov
R=results/spark/sr_ucf290_d3nano_ft/20261109-141810
A=results/pro6000/wise_a5
F=$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
M=results/spark/frames_ucf290_f8/20261105-180529/manifest.json
E=results/spark/tgm_size_ucf290/20261105-200402/jev-hard-events.jsonl
MIX=results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c

# 1. memory check: expected peak about 5 GB, need 15 GB available
avail=$(free -g | awk '/^Mem:/ {print $7}')
echo "available ${avail} GB"
[ "$avail" -ge 15 ] || { echo "not enough memory, stop"; exit 1; }

# 2. smoke test of the 2x path, 5 windows
S=results/smoke/wise_a5/$(date -u +%Y%m%d-%H%M%S)
timeout 15m uv run --frozen python scripts/sr-score.py --adapter $A --frames-dir $F --index $I --upscale bicubic2 --limit-windows 5 --out $S/B/scores.jsonl
echo "smoke ok $(date -u +%FT%TZ)"

# 3. real runs
echo "draft resume $(date -u +%FT%TZ)"
timeout 3h uv run --frozen python scripts/sr-score.py --adapter $A --frames-dir $F --index $I --upscale none --out $R/wise_a5/draft/scores.jsonl
echo "B start $(date -u +%FT%TZ)"
timeout 6h uv run --frozen python scripts/sr-score.py --adapter $A --frames-dir $F --index $I --upscale bicubic2 --out $R/wise_a5/B/scores.jsonl
echo "scoring done $(date -u +%FT%TZ)"

# 4. evaluation (CPU)
uv run --frozen python scripts/layout-eval.py --manifest $M --events $E --draft $MIX/draft/scores.jsonl \
  --run mix2x:$MIX/B/scores.jsonl wise1x:$R/wise_a5/draft/scores.jsonl wise2x:$R/wise_a5/B/scores.jsonl --out $R/eval-vs-mixkl.json 2>&1 | tee $R/eval-vs-mixkl.log
uv run --frozen python scripts/layout-eval.py --manifest $M --events $E --draft $R/wise_a5/draft/scores.jsonl \
  --run wise2x:$R/wise_a5/B/scores.jsonl --out $R/eval-2x-vs-1x.json 2>&1 | tee $R/eval-2x-vs-1x.log
uv run --frozen python results/spark/sr_ucf290_d3nano_zs/20261106-040502/scripts/score-shift.py \
  --draft $R/wise_a5/draft/scores.jsonl --cond $R/wise_a5/B/scores.jsonl --manifest $M --events $E 2>&1 | tee $R/score-shift-B.log
uv run --frozen python results/spark/frame_auc_ucf290/20261109-153701/frame-auc.py $M $R/frame-auc.json \
  mix1x=$MIX/draft/scores.jsonl mix2x=$MIX/B/scores.jsonl wise1x=$R/wise_a5/draft/scores.jsonl wise2x=$R/wise_a5/B/scores.jsonl 2>&1 | tee $R/frame-auc.log
echo "all done $(date -u +%FT%TZ)"
