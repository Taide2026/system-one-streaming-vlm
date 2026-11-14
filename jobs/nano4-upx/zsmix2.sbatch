#!/bin/bash
#SBATCH --job-name=s1-zsmix2
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=06:00:00
#SBATCH --output=/work/u9117820/s1-logs/zsmix2-%j.out
# Zero-shot d3-nano, mixed tokens: fine tokens on every block the motion region touches, no margin (2026-11-12).
# The thr-0.25 variant of zsmix.sbatch leaves small motion regions with no fine block at all; this is the intended "tight" one.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
name=up290_zs_mixdrop_touch; mkeval $name "" frames > /dev/null
$PY scripts/mech.py pool --config configs/$name.yaml --mode mixdrop --kernel bicubic --thr 0.05 --dilate 0 --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) 2>&1 | grep --line-buffered -vE 'Warning|Loading weights|causal_conv1d'
echo ZSMIX2_DONE
