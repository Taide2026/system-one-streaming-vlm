#!/bin/bash
#SBATCH --job-name=s1-bllat
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=01:00:00
#SBATCH --output=/work/u9117820/s1-logs/bllat-%j.out
# Same-GPU latency of the baseline layouts (2026-11-13), on its own GPU so it does not wait for the evaluations.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
for k in ref fla; do
  $PY scripts/bl.py latency --config configs/blself.yaml --out results/nano4/latbl/$SLURM_JOB_ID-$k.json --kernels $k 2>&1 | grep -E '^(reference|fla) |BL_LAT_DONE|Error|Traceback' || true
done
echo BLLAT_DONE
