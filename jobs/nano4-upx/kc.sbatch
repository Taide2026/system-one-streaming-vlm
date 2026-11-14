#!/bin/bash
#SBATCH --job-name=s1-kc
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/kc-%j.out
# Keyframe + motion crops, the cost version (2026-11-13), full UCF-290. sbatch --export=ALL,MODEL=zs|floor_s0 kc.sbatch
#   kc1  key frame 3      + crop of the touch region in the other 7 frames
#   kc2  key frame 3      + crop of the loose region
#   kc3  key frames 0, 7  + crop of the touch region in the other 6
#   kc4  key frames 0, 7  + crop of the loose region
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs; AD=""; [ "$MODEL" = floor_s0 ] && { fetch $F0; AD=$F0; }
run() {  # $1 tag, then keycrop.py flags
  local name=up290_${MODEL}_$1; shift
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name "$AD" frames > /dev/null
  $PY scripts/keycrop.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|KEYCROP_DONE|Error|Traceback' && echo "DONE $name"
}
export -f run; export MODEL AD
CUDA_VISIBLE_DEVICES=0 run kc1 --key 3 --rest touch > $L/kc-$MODEL-1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 run kc2 --key 3 --rest loose > $L/kc-$MODEL-2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 run kc3 --key 0 7 --rest touch > $L/kc-$MODEL-3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 run kc4 --key 0 7 --rest loose > $L/kc-$MODEL-4.log 2>&1 &
wait; echo KC_JOB_DONE
