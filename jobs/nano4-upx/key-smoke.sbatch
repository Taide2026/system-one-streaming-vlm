#!/bin/bash
#SBATCH --job-name=s1-ksmoke
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/work/u9117820/s1-logs/ksmoke-%j.out
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
rm -rf results/nano4/key-smoke; mkeval key_smoke "" frames > /dev/null
i=0
for f in "--key 3 --key-layout fine --rest touch" "--key 0 7 --key-layout fine --rest touch" "--key 3 --key-layout coarse --rest touch" "--key --rest touch" "--key 3 --key-layout fine --rest touch --rest-coarse 1" "--key 0 1 2 3 4 5 6 7 --key-layout fine --rest none"; do
  i=$((i+1)); $PY scripts/keyframe.py --config configs/key_smoke.yaml --out results/nano4/key-smoke/c$i --limit 60 $f 2>&1 | grep -E 'KEY_DONE|Error|Traceback' -A6 || true
  $PY -c "
import json,numpy as np
R=[json.loads(l) for l in open('results/nano4/key-smoke/c$i/scores.jsonl')]
print('c$i [$f] windows',len(R),'tokens/window',round(np.mean([r['tokens'] for r in R])),'mean margin %.3f'%np.mean([r['margin'] for r in R]))"
done
echo KSMOKE_DONE
