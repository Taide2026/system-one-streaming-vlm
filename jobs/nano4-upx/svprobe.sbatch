#!/bin/bash
#SBATCH --job-name=s1-svprobe
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/work/u9117820/s1-logs/svprobe-%j.out
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
for m in qwen3-vl-4b internvl3.5-2b smolvlm2-2.2b gemma-3-4b gemma-4-e4b; do
  $PY scripts/smallvlm.py --model $W/models-ext/$m --config configs/floor290_d3nano_c.yaml --out /dev/null --probe 3 2>&1 | grep -E 'PROBE|options|Error|Traceback|error' | cut -c1-220 || echo "FAILED $m"
done
echo SVPROBE_DONE
