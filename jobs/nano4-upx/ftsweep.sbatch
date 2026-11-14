#!/bin/bash
#SBATCH --job-name=s1-ftsweep
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:3
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=10:00:00
#SBATCH --output=/work/u9117820/s1-logs/ftsweep-%j.out
# Whole-frame bicubic 1.25 / 1.5 / 1.75x on the fine-tuned model, seeds 0-2, full UCF-290 (2026-11-13): the cost curve
# the keyframe method has to beat. One GPU per seed, the three factors one after the other (1.5x first).
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs; fetch $F0 $F1 $F2; A=($F0 $F1 $F2)
for s in 0 1 2; do
  ( export CUDA_VISIBLE_DEVICES=$s; ev up290_floor_s${s}_1p5 ${A[$s]} up:1.5; ev up290_floor_s${s}_1p25 ${A[$s]} up:1.25; ev up290_floor_s${s}_1p75 ${A[$s]} up:1.75 ) > $L/ftsweep-s$s.log 2>&1 &
done
wait; echo FTSWEEP_DONE
