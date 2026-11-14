#!/bin/bash
#SBATCH --job-name=s1-xsmoke
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/work/u9117820/s1-logs/xsmoke-%j.out
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
rm -rf results/nano4/mix-smoke; mkdir -p results/nano4/mix-smoke
for m in identity mixrep mixdrop mixdrop_rand dropcoarse pool1x; do
  $PY scripts/mech.py pool --config configs/mech_smoke.yaml --mode $m --kernel bicubic --out results/nano4/mix-smoke/$m 2>&1 | grep --line-buffered -vE 'Warning|Loading weights|causal_conv1d' | tail -2
done
$PY - <<'PY'
import json, numpy as np
L = lambda m: [json.loads(l) for l in open(f"results/nano4/mix-smoke/{m}/scores.jsonl")]
ref = L("identity")
for m in ("mixrep", "mixdrop", "mixdrop_rand", "dropcoarse", "pool1x"):
    R = L(m); d = np.array([a["margin"] - b["margin"] for a, b in zip(R, ref)])
    print(f"{m:13s} windows {len(R)}  mean |margin - untouched 2x| {np.abs(d).mean():.3f}  tokens/window {np.mean([r.get('tokens', 0) for r in R]):.0f}  fine blocks {np.mean([r.get('fine_blocks', 0) for r in R]):.1f} of {R[0].get('blocks', 0)}")
PY
echo XSMOKE_DONE
