#!/bin/bash
#SBATCH --job-name=s1-psmoke
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/work/u9117820/s1-logs/psmoke-%j.out
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
rm -rf results/nano4/prune-smoke; mkeval prune_smoke "" frames > /dev/null
for f in "--budget 0" "--budget 822 --rule attention" "--budget 822 --rule random"; do
  d=results/nano4/prune-smoke/$(echo $f | tr -d ' -')
  $PY scripts/prune.py --config configs/prune_smoke.yaml --out $d --limit 40 $f 2>&1 | grep -vE 'Warning|Loading weights|causal_conv1d' | tail -8
done
echo PSMOKE_DONE
