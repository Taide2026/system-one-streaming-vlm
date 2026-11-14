#!/bin/bash
#SBATCH --job-name=s1-prune
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/prune-%j.out
# FastV-style pruning baseline on the 2x input (2026-11-13), full UCF-290: zero-shot and fine-tuned seed 0, visual
# tokens cut to 822 (the budget of "touch") after the first softmax-attention layer, by attention and at random.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs; fetch $F0
run() {  # $1 model tag, $2 adapter, $3 rule
  local name=up290_$1_prune822_$3
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name "$2" frames > /dev/null
  $PY scripts/prune.py --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) --budget 822 --rule $3 2>&1 | grep --line-buffered -E 'window|PRUNE_DONE|Error|Traceback' && echo "DONE $name"
}
export -f run
CUDA_VISIBLE_DEVICES=0 run zs "" attention > $L/prune-zs-att.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 run zs "" random > $L/prune-zs-rand.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 run floor_s0 $F0 attention > $L/prune-ft-att.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 run floor_s0 $F0 random > $L/prune-ft-rand.log 2>&1 &
wait; echo PRUNE_JOB_DONE
