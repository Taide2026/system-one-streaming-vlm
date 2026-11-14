#!/bin/bash
#SBATCH --job-name=s1-zsmix
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/zsmix-%j.out
# Basic set 2 (2026-11-12): zero-shot d3-nano, "2x effect at 1x cost". 2x frames through the vision tower; per frame one
# coarse token per 2x2 block (the block mean) and the fine 2x tokens only on the TGM motion region. Full UCF-290.
#   dropcoarse    coarse tokens only (640 per window, the 1x cost)
#   mixdrop_tight fine tokens on the blocks the motion region covers by more than a quarter (no margin)
#   mixdrop       fine tokens on every block the region touches plus the blocks around them
#   mixdrop_rand  as mixdrop, the same number of fine blocks at random places (does the location matter?)
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs
pool() {  # $1 name tag, $2 mode, then extra flags
  local name=up290_zs_$1 mode=$2; shift 2
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name "" frames > /dev/null
  $PY scripts/mech.py pool --config configs/$name.yaml --mode $mode --kernel bicubic "$@" --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) 2>&1 | grep --line-buffered -vE 'Warning|Loading weights|causal_conv1d' && echo "DONE $name"
}
export -f pool
CUDA_VISIBLE_DEVICES=0 pool dropcoarse dropcoarse > $L/zsmix-dropcoarse.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 pool mixdrop_tight mixdrop --thr 0.25 --dilate 0 > $L/zsmix-mixdrop_tight.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 pool mixdrop mixdrop > $L/zsmix-mixdrop.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 pool mixdrop_rand mixdrop_rand > $L/zsmix-mixdrop_rand.log 2>&1 &
wait; echo ZSMIX_DONE
