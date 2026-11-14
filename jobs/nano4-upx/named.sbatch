#!/bin/bash
#SBATCH --job-name=s1-named
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:6
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=12:00:00
#SBATCH --output=/work/u9117820/s1-logs/named-%j.out
# Four named baselines as separate rows (2026-11-14): FastV, VisionZip, RLT (rules of the official code), CodecSight (no
# code released: our implementation). All eight frames enlarged 2x; each method uses its own signal. Each at its published
# default (FastV R=50% -> 1200 tokens; VisionZip 65%+5% -> 1680; RLT tau 0.1; CodecSight 0.25 px) and, for the two with a
# budget, at the 1015 tokens of the rectangle. UCF-Crime 290 x three seeds, ComplexVAD held-out 83 and MSAD 221 x
# zero-shot / UCF fine-tune. Then latency with both kernels in one process each.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; fetch $F0 $F1 $F2
mkeval blself $F0 frames > /dev/null
CUDA_VISIBLE_DEVICES=0 $PY scripts/bl.py selftest --config configs/blself.yaml --out /dev/null 2>&1 | grep -E 'nothing dropped|fastv|visionzip|rlt|codecsight|SELFTEST|Error|Traceback' | cut -c1-160 | tee $L/named-$J-self.log
grep -q SELFTEST_OK $L/named-$J-self.log || { echo "selftest failed"; exit 1; }
done_already() { for d in results/nano4/$1/*/; do [ -f $d/run.json ] && return 0; done 2>/dev/null; return 1; }
r() {  # $1 name, $2 adapter, $3 template ("" = UCF-290), then bl.py flags
  local name=$1 ad=$2 tpl=$3; shift 3; done_already $name && { echo "DONE $name (already)"; return 0; }
  mkeval $name "$ad" frames $tpl > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E '0000 windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
row() {  # $1 tag, then bl.py flags: MSAD first (short), then ComplexVAD, then UCF-Crime
  local t=$1; shift
  r msad_zs_v180_nb_$t "" configs/msad_tpl_v180.yaml "$@"; r msad_f0_v180_nb_$t $F0 configs/msad_tpl_v180.yaml "$@"
  r cvad83_zs_v180_nb_$t "" configs/cvad_tpl_v180_83.yaml "$@"; r cvad83_f0_v180_nb_$t $F0 configs/cvad_tpl_v180_83.yaml "$@"
  r up290_floor_s0_nb_$t $F0 "" "$@"; r up290_floor_s1_nb_$t $F1 "" "$@"; r up290_floor_s2_nb_$t $F2 "" "$@"
}
export -f done_already r row
( export CUDA_VISIBLE_DEVICES=0; row fastv1200 --mode fastv --budget 1200 ) > $L/named-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; row fastv1015 --mode fastv --budget 1015 ) > $L/named-$J-g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; row vz1680 --mode visionzip --budget 1680 ) > $L/named-$J-g2.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; row vz1015 --mode visionzip --budget 1015 ) > $L/named-$J-g3.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=4; row rlt --mode rlt --tau 0.1 ) > $L/named-$J-g4.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=5; row codecsight --mode codecsight --mv-tau 0.25 ) > $L/named-$J-g5.log 2>&1 &
wait; grep -h -c '^DONE' $L/named-$J-g*.log; grep -h -E 'FAILED|Traceback|Error' $L/named-$J-g*.log | sort | uniq -c | head
# latency, UCF-Crime fine-tuned seed 0, alone on the node's GPU 0: the named rows and two anchors from the earlier table
mkeval nblat $F0 frames > /dev/null
ONLY=("rect in place" "patch diff 1015" "fastv 1200" "fastv 1015" "visionzip 1680" "visionzip 1015" "rlt tau .1" "codecsight .25")
for kern in fla ref; do
  CUDA_VISIBLE_DEVICES=0 $PY scripts/bl.py latency --config configs/nblat.yaml --out results/nano4/nblat/lat-$kern.json --kernels $kern --only "${ONLY[@]}" 2>&1 | grep -E 'total p50|BL_LAT_DONE|Error|Traceback' | cut -c1-200
done
for t in fastv1200 fastv1015 vz1680 vz1015 rlt codecsight; do
  echo "== $t"; $PY scripts/cvad-metrics.py msad_zs_v180up2 msad_zs_v180_nb_$t | cut -c1-170 | tail -1; $PY scripts/cvad-metrics.py msad_f0_v180up2 msad_f0_v180_nb_$t | cut -c1-170 | tail -1
done
echo NAMED_DONE
