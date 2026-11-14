#!/bin/bash
#SBATCH --job-name=s1-kcsmoke
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/work/u9117820/s1-logs/kcsmoke-%j.out
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
rm -rf results/nano4/kc-smoke; mkeval kc_smoke "" frames > /dev/null
$PY scripts/keycrop.py eval --config configs/kc_smoke.yaml --out results/nano4/kc-smoke/e1 --key 0 7 --rest touch --limit 80 2>&1 | grep -vE 'Warning|Loading weights|causal_conv1d' | tail -12
$PY -c "
import json,numpy as np
R=[json.loads(l) for l in open('results/nano4/kc-smoke/e1/scores.jsonl')]
print('windows',len(R),'tokens/window',round(np.mean([r['tokens'] for r in R])),'patches/window',round(np.mean([r['patches'] for r in R])),'mean margin %.3f'%np.mean([r['margin'] for r in R]))"
$PY scripts/keycrop.py latency --config configs/kc_smoke.yaml --out results/nano4/kc-smoke/lat.json --kernels fla --limit 60 2>&1 | grep -E '^(reference|fla)|Error|Traceback' -A8 | head -30
echo KCSMOKE_DONE
