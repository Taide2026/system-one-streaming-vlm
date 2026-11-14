#!/bin/bash
#SBATCH --job-name=s1-kclat
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=03:00:00
#SBATCH --output=/work/u9117820/s1-logs/kclat-%j.out
# Latency of the keyframe + crop layouts next to whole-frame 1x / 1.5x / 2x and the token-reduced layouts, all on the
# same GPU in one job (2026-11-13): zero-shot, reference kernels then fast kernels; then the fine-tuned model, reference.
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
fetch $F0
mkeval lat_zs "" frames > /dev/null; mkeval lat_floor_s0 $F0 frames > /dev/null
T=$(date +%Y%m%d-%H%M%S)
for mk in zs:ref zs:fla floor_s0:ref; do m=${mk%%:*}; k=${mk#*:}
  $PY scripts/lat-mix.py --config configs/lat_$m.yaml --kernels $k --out results/nano4/kclat/$T/${m}_${k}_whole.json 2>&1 | grep --line-buffered -E '^(reference|fla)|Error|Traceback' || echo "FAILED whole $m $k"
  $PY scripts/keycrop.py latency --config configs/lat_$m.yaml --kernels $k --out results/nano4/kclat/$T/${m}_${k}_key.json 2>&1 | grep --line-buffered -E '^(reference|fla)|Error|Traceback' || echo "FAILED key $m $k"
done
echo KCLAT_DONE
