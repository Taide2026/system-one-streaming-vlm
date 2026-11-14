#!/bin/bash
#SBATCH --job-name=s1-kc2
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/kc2-%j.out
# Cheaper keyframe layouts (2026-11-13): the context at the ORIGINAL resolution, 2x only on the moving region.
# sbatch --export=ALL,MODEL=zs|floor_s0 kc2.sbatch          full UCF-290
#   kc5  key frames 0, 7 at 1x + 2x crop of the touch region in the other 6 frames
#   kc6  all 8 frames at 1x + 2x crop of the touch region laid over each (the cost version of "touch")
#   kc7  key frames 0, 7 at 1x + 2x crop of the loose region
#   kc8  all 8 frames at 1x + 2x crop of the loose region
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs; AD=""; [ "$MODEL" = floor_s0 ] && { fetch $F0; AD=$F0; }
mkeval kc2_check_$MODEL "$AD" frames > /dev/null; rm -rf results/nano4/kc2-check-$MODEL
CUDA_VISIBLE_DEVICES=0 $PY scripts/keycrop.py eval --config configs/kc2_check_$MODEL.yaml --out results/nano4/kc2-check-$MODEL/a --key 0 1 2 3 4 5 6 7 --key-scale 1 --crop-all 1 --rest touch --limit 30 2>&1 | grep -E 'KEYCROP_DONE|Error|Traceback' -A8
[ -f results/nano4/kc2-check-$MODEL/a/run.json ] || { echo "check failed"; exit 1; }
run() {  # $1 tag, then keycrop.py flags
  local name=up290_${MODEL}_$1; shift
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name "$AD" frames > /dev/null
  $PY scripts/keycrop.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|KEYCROP_DONE|Error|Traceback' && echo "DONE $name"
}
export -f run; export MODEL AD
CUDA_VISIBLE_DEVICES=0 run kc5 --key 0 7 --key-scale 1 --rest touch > $L/kc2-$MODEL-5.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 run kc6 --key 0 1 2 3 4 5 6 7 --key-scale 1 --crop-all 1 --rest touch > $L/kc2-$MODEL-6.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 run kc7 --key 0 7 --key-scale 1 --rest loose > $L/kc2-$MODEL-7.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 run kc8 --key 0 1 2 3 4 5 6 7 --key-scale 1 --crop-all 1 --rest loose > $L/kc2-$MODEL-8.log 2>&1 &
wait; echo KC2_JOB_DONE
