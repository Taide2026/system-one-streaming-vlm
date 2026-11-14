#!/bin/bash
#SBATCH --job-name=s1-bl
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=05:00:00
#SBATCH --output=/work/u9117820/s1-logs/bl-%j.out
# Baselines against the key-frame + motion-crop layout (2026-11-13), fine-tuned d3-nano, full UCF-290, three seeds:
# whole 1x rest (LoHi style), pooled rest (SlowFast style), feature-level differential token drop, patch drop before
# the vision tower (our mask; pixel change at a matched budget), crop without enlarging. Then same-GPU latency.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; fetch $F0 $F1 $F2
mkeval blself "" frames > /dev/null
CUDA_VISIBLE_DEVICES=0 $PY scripts/bl.py selftest --config configs/blself.yaml --out /dev/null 2>&1 | grep -v -E 'Warning|warn|falling back' | tail -14
bl() {  # $1 seed, $2 adapter, $3 tag, then flags
  local name=up290_floor_s$1_bl_$3 ad=$2; shift 3
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name $ad frames > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
export -f bl
LIST=()
for m in "pool --mode pool" "dtd --mode dtd" "pdiff --mode patch-diff" "ptgm --mode patch-tgm" "w1x --mode whole1x" "crop1 --mode crop --crop-scale 1"; do
  LIST+=("0 $F0 $m" "1 $F1 $m" "2 $F2 $m")
done
LIST+=("0 $F0 prlt --mode patch-diff --key 0 --tau 0.02" "0 $F0 cropchk --mode crop")
G=$(nvidia-smi -L | wc -l); echo "${#LIST[@]} evaluations on $G GPUs"
for g in $(seq 0 $((G - 1))); do
  ( i=0; for job in "${LIST[@]}"; do
      if [ $((i % G)) -eq $g ]; then CUDA_VISIBLE_DEVICES=$g bl $job || echo "FAILED $job"; fi; i=$((i + 1))
    done ) > $L/bl-$J-g$g.log 2>&1 &
done
wait; grep -h -E '^DONE|FAILED|Traceback|Error' $L/bl-$J-g*.log || true
for k in ref fla; do
  CUDA_VISIBLE_DEVICES=0 $PY scripts/bl.py latency --config configs/blself.yaml --out results/nano4/latbl/$J-$k.json --kernels $k 2>&1 | grep -E '^(reference|fla) |BL_LAT_DONE|Error|Traceback' || true
done
echo BL_ALL_DONE
