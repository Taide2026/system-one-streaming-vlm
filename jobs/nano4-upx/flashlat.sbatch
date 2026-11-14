#!/bin/bash
#SBATCH --job-name=s1-flashlat
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=02:00:00
#SBATCH --output=/work/u9117820/s1-logs/flashlat-%j.out
# Latency by stage for d3-flash (2026-11-13), zero-shot, one GPU: whole-frame factors and token cuts (lat-mix.py),
# key frames + crops (keycrop.py), with reference and with fast kernels.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
mkeval lat_flash_zs "" frames configs/floor290_d3flash_c_s1.yaml > /dev/null
for k in ref fla; do
  $PY scripts/lat-mix.py --config configs/lat_flash_zs.yaml --kernels $k --out results/nano4/latflash/$SLURM_JOB_ID/whole_$k.json 2>&1 | grep --line-buffered -E '^(reference|fla)|LAT_DONE|Error|Traceback' || echo "FAILED whole $k"
  $PY scripts/keycrop.py latency --config configs/lat_flash_zs.yaml --kernels $k --out results/nano4/latflash/$SLURM_JOB_ID/key_$k.json 2>&1 | grep --line-buffered -E '^(reference|fla)|KEYCROP_LAT_DONE|Error|Traceback' -A6 || echo "FAILED key $k"
done
echo FLASHLAT_DONE
