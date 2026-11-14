#!/bin/bash
#SBATCH --job-name=s1-mixq
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=03:00:00
#SBATCH --output=/work/u9117820/s1-logs/mixq-%j.out
# Quick go/no-go (2026-11-12): mixed-resolution tokens on the 103-video screening subset, seed 0 only.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs; fetch $F0
pool() {  # $1 mode
  local name=scr_floor_s0_$1
  mkeval $name $F0 frames > /dev/null
  $PY - $name <<'PY'
import json, sys, yaml
n = json.load(open("configs/screen_ucf_names.json")); n = n["names"] if isinstance(n, dict) else n
n = [x["name"] if isinstance(x, dict) else x for x in n]
d = yaml.safe_load(open(f"configs/{sys.argv[1]}.yaml")); d["dataset"] = {"name": d["dataset"]["name"], "asset": d["dataset"]["asset"], "names": n}
yaml.safe_dump(d, open(f"configs/{sys.argv[1]}.yaml", "w"), sort_keys=False); print(sys.argv[1], len(n), "videos")
PY
  $PY scripts/mech.py pool --config configs/$name.yaml --mode $1 --kernel bicubic --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) 2>&1 | grep --line-buffered -vE 'Warning|Loading weights|causal_conv1d' && echo "DONE $name"
}
export -f pool
i=0; for m in mixdrop mixdrop_tight mixdrop_rand dropcoarse; do CUDA_VISIBLE_DEVICES=$i pool $m > $L/mixq-$m.log 2>&1 & i=$((i+1)); done
wait; echo MIXQ_DONE
