#!/bin/bash
#SBATCH --job-name=s1-nself
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/work/u9117820/s1-logs/nself-%j.out
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
fetch $F0
mkeval blself $F0 frames > /dev/null
$PY scripts/bl.py selftest --config configs/blself.yaml --out /dev/null 2>&1 | tail -48
echo NSELF_DONE
