#!/bin/bash
#SBATCH --job-name=s1-mech2
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn009
#SBATCH --time=03:00:00
#SBATCH --output=/work/u9117820/s1-logs/mech2-%j.out
# Layer-wise comparison of 1x, bicubic 2x and nearest 2x on a 300-window probe set (2026-11-12), seed 0.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
A=$(ls -d results/nano4/up290_floor_s0_frames/*/ | tail -1)
OUT=results/nano4/mech/layers-$(date +%Y%m%d-%H%M%S)
$PY scripts/mech.py layers --config configs/viz_floor.yaml --events results/pro6000/jev-hard-events.jsonl --a $A --out $OUT 2>&1 | grep --line-buffered -vE 'Warning|Loading weights|causal_conv1d'
echo $OUT; echo MECH2_DONE
