#!/bin/bash
#SBATCH --job-name=s1-blind
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=04:00:00
#SBATCH --output=/work/u9117820/s1-logs/blind-%j.out
# The blind spot of change-driven input selection (2026-11-13), ComplexVAD, the fixed 30-video list, 180p base, 2x layout.
# (1) CPU: how much of the annotated boxes each selection covers (cvad-cover.py).
# (2) GPU: other people's selections on the same videos, at the crop's token budget (750 per window):
#     patch-diff / patch-tgm (before the tower, change-driven), dtd (after the tower, change-driven),
#     pool (after the tower, NOT change-driven: every position kept, coarser) as the control.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; ROOT=$W/datasets/complexvad; fetch $F0
latest() { ls -d results/nano4/$1/*/ | while read d; do [ -f $d/run.json ] && echo $d; done | tail -1; }
( $PY scripts/cvad-cover.py $ROOT --list list30.txt --scores-a $(latest cvad30_zs_v180up2)scores.jsonl --scores-b $(latest cvad30_zs_v180kc3)scores.jsonl --out results/nano4/cvad-cover-30.jsonl 2>&1 | grep -v -E '^[0-9]+ windows so far|Warning' | tail -30 ) > $L/blind-$J-cover.log 2>&1 &
b() {  # $1 model tag, $2 adapter, $3 name, then flags
  local name=cvad30_$1_v180_$3 ad=$2; shift 3
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name "$ad" frames configs/cvad_tpl_v180_30.yaml > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
export -f b
grep -q 'videos: list30.txt' configs/cvad_tpl_v180_30.yaml || { echo "template does not pin the 30-video list"; exit 1; }
( export CUDA_VISIBLE_DEVICES=0; b zs "" pdiff --mode patch-diff --budget 750; b f0 $F0 pdiff --mode patch-diff --budget 750 ) > $L/blind-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; b zs "" ptgm --mode patch-tgm; b f0 $F0 ptgm --mode patch-tgm ) > $L/blind-$J-g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; b zs "" dtd --mode dtd --budget 750; b f0 $F0 dtd --mode dtd --budget 750 ) > $L/blind-$J-g2.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; b zs "" pool --mode pool; b f0 $F0 pool --mode pool ) > $L/blind-$J-g3.log 2>&1 &
wait; cat $L/blind-$J-cover.log; grep -h -E '^DONE|FAILED|Traceback|Error' $L/blind-$J-g*.log || true
for m in zs f0; do $PY scripts/cvad-metrics.py cvad30_${m}_v180up2 cvad30_${m}_v180kc3 cvad30_${m}_v180_ptgm cvad30_${m}_v180_pdiff cvad30_${m}_v180_dtd cvad30_${m}_v180_pool | cut -c1-190; done
echo BLIND_DONE
