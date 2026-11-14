#!/bin/bash
#SBATCH --job-name=s1-sh290
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=06:00:00
#SBATCH --output=/work/u9117820/s1-logs/sh290-%j.out
# Third model (2026-11-13): Shieldstral 1.0 3B (Mistral; Pixtral encoder + Ministral 3B), zero-shot, full UCF-290.
# Whole frame 1x, 1.5x, 2x and key frames + motion crops sent as plain images (input form keycrop).
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID
one() {  # $1 name, $2 input_form yaml body
  for d in results/nano4/$1/*/; do [ -f $d/run.json ] && { echo "$1 done already"; return 0; }; done 2>/dev/null
  $PY - "$1" "$2" <<'PYEOF'
import sys, yaml
name, block = sys.argv[1:]
d = yaml.safe_load(open("configs/floor290_d3nano_c.yaml")); d["name"] = name
d["backbone"] = {"name": "shieldstral", "asset": "shieldstral-3b", "memory_cap_gb": 40}
d["input_form"] = yaml.safe_load(block)
yaml.safe_dump(d, open(f"configs/{name}.yaml", "w"), sort_keys=False)
PYEOF
  $PY -m s1stream.cli run configs/$1.yaml --out results/nano4/$1/$(date +%Y%m%d-%H%M%S) && echo "DONE $1"
}
export -f one
CUDA_VISIBLE_DEVICES=0 one sh290_zs_2x  "$(printf 'name: upsample\nfactor: 2\nkernel: bicubic\n')" > $L/sh290-$J-g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 one sh290_zs_1p5 "$(printf 'name: upsample\nfactor: 1.5\nkernel: bicubic\n')" > $L/sh290-$J-g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 one sh290_zs_1x  "name: frames" > $L/sh290-$J-g2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 one sh290_zs_kc  "$(printf 'name: keycrop\nkey: [0, 7]\nfactor: 2\n')" > $L/sh290-$J-g3.log 2>&1 &
wait; grep -h -E '^DONE|FAILED|Traceback|Error' $L/sh290-$J-g*.log || true; echo SH290_DONE
