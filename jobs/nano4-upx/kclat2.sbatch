#!/bin/bash
#SBATCH --job-name=s1-kclat2
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=03:00:00
#SBATCH --output=/work/u9117820/s1-logs/kclat2-%j.out
# Latency of all eight keyframe layouts (2x and 1x key frames) next to whole-frame 1x / 1.5x / 2x, same GPU, zero-shot,
# reference then fast kernels (2026-11-13).
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
mkeval lat_zs "" frames > /dev/null
T=$(date +%Y%m%d-%H%M%S)
for k in ref fla; do
  $PY scripts/lat-mix.py --config configs/lat_zs.yaml --kernels $k --out results/nano4/kclat2/$T/zs_${k}_whole.json 2>&1 | grep --line-buffered -E '^(reference|fla)|Error|Traceback' || echo "FAILED whole $k"
  $PY scripts/keycrop.py latency --config configs/lat_zs.yaml --kernels $k --out results/nano4/kclat2/$T/zs_${k}_key.json 2>&1 | grep --line-buffered -E '^(reference|fla)|Error|Traceback' -A6 || echo "FAILED key $k"
done
echo KCLAT2_DONE
