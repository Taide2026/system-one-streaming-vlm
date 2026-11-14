#!/bin/bash
#SBATCH --job-name=s1-mech1c
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=12:00:00
#SBATCH --output=/work/u9117820/s1-logs/mech1c-%j.out
# Token-pooling intervention as full UCF-290 evaluations (2026-11-12), original-trained floor, seeds 0-2:
#   pool1x  : 2x bicubic frames through the vision tower, tokens pooled to the 1x grid (features 2x, count 1x)
#   poolrep : the pooled tokens replicated to the 2x grid (count 2x, no content finer than the 1x grid)
# plus the ordinary nearest-neighbour 2x input for the three seeds, and pool1x on nearest for seed 0.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs
fetch $F0 $F1 $F2
pool() {  # $1 seed, $2 adapter, $3 mode, $4 kernel
  local name=up290_floor_s$1_$3; [ $4 = bicubic ] || name=${name}_$4
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name $2 frames > /dev/null
  $PY scripts/mech.py pool --config configs/$name.yaml --mode $3 --kernel $4 --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) 2>&1 | grep --line-buffered -vE 'Warning|Loading weights|causal_conv1d' && echo "DONE $name"
}
export -f pool
A=($F0 $F1 $F2)
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$s pool $s ${A[$s]} pool1x bicubic > $L/mech1c-pool1x-s$s.log 2>&1 & done
( export CUDA_VISIBLE_DEVICES=3; ev up290_floor_s0_near2x $F0 up:2:nearest; ev up290_floor_s2_near2x $F2 up:2:nearest ) > $L/mech1c-g6.log 2>&1 &
wait; echo MECH1A_DONE
