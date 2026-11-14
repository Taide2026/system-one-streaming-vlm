#!/bin/bash
#SBATCH --job-name=s1-flash
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=10:00:00
#SBATCH --output=/work/u9117820/s1-logs/flash-%j.out
# Second model (2026-11-13): d3-flash (8.4B, the same family with a four times larger language model), full UCF-290.
# Zero-shot and the two fine-tuned seeds that exist (floor_d3flash_c_s1, _s2; their 1x evaluations are
# floor290_d3flash_c_s1/_s2). Whole frame 2x and key frames + motion crops for all three; on seed 1 also whole frame
# 1.5x and the two token cuts after the vision tower (by motion, at random).
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; T=configs/floor290_d3flash_c_s1.yaml
S1=results/nano4/floor_d3flash_c_s1/20261107-044624; S2=results/nano4/floor_d3flash_c_s2/20261107-044624
done_already() { for d in results/nano4/$1/*/; do [ -f $d/run.json ] && return 0; done 2>/dev/null; return 1; }
whole() { ev fl290_$1_$3 "$2" $4 $T; }                                  # $1 tag, $2 adapter, $3 name, $4 form
kc() {
  local name=fl290_$1_kc3; done_already $name && return 0
  mkeval $name "$2" frames $T > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) --mode crop --key 0 7 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
touch_() {
  local name=fl290_$1_touch; done_already $name && return 0
  mkeval $name "$2" frames $T > /dev/null
  $PY scripts/mech.py pool --config configs/$name.yaml --mode mixdrop --kernel bicubic --thr 0.05 --dilate 0 --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) 2>&1 | grep --line-buffered -E 'windows|POOL_DONE|Error|Traceback' && echo "DONE $name"
}
prand() {
  local name=fl290_$1_prand; done_already $name && return 0
  mkeval $name "$2" frames $T > /dev/null
  $PY scripts/prune.py --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) --budget 822 --rule random 2>&1 | grep --line-buffered -E 'windows|PRUNE_DONE|Error|Traceback' && echo "DONE $name"
}
export -f done_already whole kc touch_ prand; export T S1 S2
( export CUDA_VISIBLE_DEVICES=0; whole zs "" 2x up:2;  whole zs "" 1x frames; kc zs "" ) > $L/flash-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; whole s1 $S1 2x up:2; kc s1 $S1 ) > $L/flash-$J-g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; whole s2 $S2 2x up:2; kc s2 $S2 ) > $L/flash-$J-g2.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; touch_ s1 $S1; prand s1 $S1; whole s1 $S1 1p5 up:1.5 ) > $L/flash-$J-g3.log 2>&1 &
wait; grep -h -E '^DONE|FAILED|Traceback|Error' $L/flash-$J-g*.log || true; echo FLASH_DONE
