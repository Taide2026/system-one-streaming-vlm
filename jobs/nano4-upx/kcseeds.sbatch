#!/bin/bash
#SBATCH --job-name=s1-kcseeds
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/kcseeds-%j.out
# Seeds 1 and 2 of the fine-tuned model for the two keyframe-crop layouts that hold up on seed 0 (2026-11-13):
# kc3 (key frames 0, 7 at 2x + touch crops) and kc4 (+ loose crops). Full UCF-290.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; fetch $F1 $F2
run() {  # $1 seed, $2 adapter, $3 tag, then flags
  local name=up290_floor_s$1_$3 ad=$2; shift 3
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name $ad frames > /dev/null
  $PY scripts/keycrop.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|KEYCROP_DONE|Error|Traceback' && echo "DONE $name"
}
export -f run
CUDA_VISIBLE_DEVICES=0 run 1 $F1 kc3 --key 0 7 --rest touch > $L/kcseeds-s1-3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 run 2 $F2 kc3 --key 0 7 --rest touch > $L/kcseeds-s2-3.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 run 1 $F1 kc4 --key 0 7 --rest loose > $L/kcseeds-s1-4.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 run 2 $F2 kc4 --key 0 7 --rest loose > $L/kcseeds-s2-4.log 2>&1 &
wait; echo KCSEEDS_DONE
