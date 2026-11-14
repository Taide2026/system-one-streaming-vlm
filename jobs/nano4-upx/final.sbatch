#!/bin/bash
#SBATCH --job-name=s1-final
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:3
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=06:00:00
#SBATCH --output=/work/u9117820/s1-logs/final-%j.out
# 2026-11-13 morning. GPU 0: latency of whole-frame 1 / 1.25 / 1.5 / 1.75 / 2x and of every keyframe layout (now with three
# key frames), one after the other on one GPU, zero-shot, reference then fast kernels. GPU 1, 2: three key frames
# (0, 3, 7) + touch crops on fine-tuned seeds 1 and 2, full UCF-290.
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs; fetch $F1 $F2; mkeval lat_zs "" frames > /dev/null; T=$(date +%Y%m%d-%H%M%S)
kc() {  # $1 seed, $2 adapter
  local name=up290_floor_s$1_kc9
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && return 0; done 2>/dev/null
  mkeval $name $2 frames > /dev/null
  $PY scripts/keycrop.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) --key 0 3 7 --rest touch 2>&1 | grep --line-buffered -E 'windows|KEYCROP_DONE|Error|Traceback' && echo "DONE $name"
}
export -f kc
CUDA_VISIBLE_DEVICES=1 kc 1 $F1 > $L/final-kc9-s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 kc 2 $F2 > $L/final-kc9-s2.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0
for k in ref fla; do
  $PY scripts/lat-mix.py --config configs/lat_zs.yaml --kernels $k --out results/nano4/latfinal/$T/zs_${k}_whole.json 2>&1 | grep --line-buffered -E '^(reference|fla)|Error|Traceback' || echo "FAILED whole $k"
  $PY scripts/keycrop.py latency --config configs/lat_zs.yaml --kernels $k --out results/nano4/latfinal/$T/zs_${k}_key.json 2>&1 | grep --line-buffered -E '^(reference|fla)|Error|Traceback' -A6 || echo "FAILED key $k"
done
wait; echo FINAL_DONE
