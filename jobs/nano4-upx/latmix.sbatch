#!/bin/bash
#SBATCH --job-name=s1-latmix
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=03:00:00
#SBATCH --output=/work/u9117820/s1-logs/latmix-%j.out
# Latency of the mixed-resolution layouts, stage by stage (2026-11-12): zero-shot and fine-tuned seed 0, each with the
# reference kernels and with the fast kernels. One GPU, the four passes one after the other so nothing shares it.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
fetch $F0
mkeval lat_zs "" frames > /dev/null; mkeval lat_floor_s0 $F0 frames > /dev/null
T=$(date +%Y%m%d-%H%M%S)
for m in zs floor_s0; do for k in ref fla; do
  $PY scripts/lat-mix.py --config configs/lat_$m.yaml --kernels $k --out results/nano4/latmix/$T/${m}_$k.json 2>&1 | grep --line-buffered -E '^(reference|fla|LAT_DONE)|Error|Traceback' || echo "FAILED $m $k"
done; done
echo LATMIX_DONE
