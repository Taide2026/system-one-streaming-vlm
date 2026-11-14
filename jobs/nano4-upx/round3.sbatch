#!/bin/bash
#SBATCH --job-name=s1-r3
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/r3-%j.out
# Round 3 of the autonomous night (2026-11-13 01:55), full UCF-290, to complete the three-seed comparison:
#   GPU 0-3  FastV-style pruning to 822 tokens (attention / random) on fine-tuned seeds 1 and 2
#   GPU 4-6  "touch" (tokens reduced after the vision tower) on fine-tuned seeds 0, 1, 2
#   GPU 7    three key frames (0, 3, 7) + touch crops, fine-tuned seed 0 then zero-shot
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs; fetch $F0 $F1 $F2; A=($F0 $F1 $F2)
done_already() { for d in results/nano4/$1/*/; do [ -f $d/run.json ] && return 0; done 2>/dev/null; return 1; }
prune() {  # $1 seed, $2 rule
  local name=up290_floor_s$1_prune822_$2; done_already $name && return 0
  mkeval $name ${A[$1]} frames > /dev/null
  $PY scripts/prune.py --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) --budget 822 --rule $2 2>&1 | grep --line-buffered -E 'windows|PRUNE_DONE|Error|Traceback' && echo "DONE $name"
}
touch_() {  # $1 seed
  local name=up290_floor_s$1_mixdrop_touch; done_already $name && return 0
  mkeval $name ${A[$1]} frames > /dev/null
  $PY scripts/mech.py pool --config configs/$name.yaml --mode mixdrop --kernel bicubic --thr 0.05 --dilate 0 --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) 2>&1 | grep --line-buffered -E 'windows|POOL_DONE|Error|Traceback' && echo "DONE $name"
}
kc() {  # $1 model tag, $2 adapter or "", $3 name tag, then flags
  local name=up290_$1_$3 ad=$2; shift 3; done_already $name && return 0
  mkeval $name "$ad" frames > /dev/null
  $PY scripts/keycrop.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|KEYCROP_DONE|Error|Traceback' && echo "DONE $name"
}
export -f done_already prune touch_ kc; export A F0 F1 F2
CUDA_VISIBLE_DEVICES=0 prune 1 attention > $L/r3-0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 prune 1 random > $L/r3-1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 prune 2 attention > $L/r3-2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 prune 2 random > $L/r3-3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 touch_ 0 > $L/r3-4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 touch_ 1 > $L/r3-5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 touch_ 2 > $L/r3-6.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=7; kc floor_s0 $F0 kc9 --key 0 3 7 --rest touch; kc zs "" kc9 --key 0 3 7 --rest touch ) > $L/r3-7.log 2>&1 &
wait; echo R3_DONE
