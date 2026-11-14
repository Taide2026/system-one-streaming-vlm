#!/bin/bash
#SBATCH --job-name=s1-zs
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:5
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/zs-%j.out
# Basic set 1 (2026-11-12): zero-shot d3-nano (no adapter), all 8 frames bicubic at 1 / 1.25 / 1.5 / 1.75 / 2x, full UCF-290.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs; i=0
for f in 1x:frames 1p25:up:1.25 1p5:up:1.5 1p75:up:1.75 2x:up:2; do
  CUDA_VISIBLE_DEVICES=$i ev up290_zs_${f%%:*} "" ${f#*:} > $L/zs-${f%%:*}.log 2>&1 & i=$((i+1))
done
wait; echo ZS_DONE
