#!/bin/bash
#SBATCH --job-name=s1-nblat
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=01:30:00
#SBATCH --output=/work/u9117820/s1-logs/nblat-%j.out
# Latency of the named baselines (2026-11-14), UCF-Crime fine-tuned seed 0, alone on one GPU, both kernels, with two
# anchors from the earlier table. (The latency step of named.sbatch did not run: a grep with no match ended that script.)
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
fetch $F0; mkeval nblat $F0 frames > /dev/null
ONLY=("rect in place" "patch diff 1015" "fastv 1200" "fastv 1015" "visionzip 1680" "visionzip 1015" "rlt tau .1" "codecsight .25")
for kern in fla ref; do
  $PY scripts/bl.py latency --config configs/nblat.yaml --out results/nano4/nblat/lat-$kern.json --kernels $kern --only "${ONLY[@]}" 2>&1 | grep -E 'total p50|BL_LAT_DONE|Error|Traceback' | cut -c1-200
done
echo NBLAT_DONE
