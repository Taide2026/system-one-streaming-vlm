#!/bin/bash
#SBATCH --job-name=s1-occl
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/u9117820/s1-logs/occl-%j.out
# 2026-11-12: attention, visual-token and occlusion probes of the fine-tuned d3-nano on original vs enlarged frames.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
A=$(ls -d results/nano4/up290_floor_s0_frames/*/ | tail -1); B=$(ls -d results/nano4/up290_floor_s0_all2x/*/ | head -1)
OUT=results/nano4/viz/occl-$(date +%Y%m%d-%H%M%S)
$PY scripts/viz-probe.py --config configs/viz_floor.yaml --events results/pro6000/jev-hard-events.jsonl --a $A --b $B --out $OUT --per-group 60 --occl-per-group 24 2>&1 | grep --line-buffered -vE 'Warning|Loading weights|causal_conv1d'
echo $OUT
