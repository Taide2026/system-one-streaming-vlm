#!/bin/bash
#SBATCH --job-name=s1-ntgm
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:3
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=12:00:00
#SBATCH --output=/work/u9117820/s1-logs/ntgm-%j.out
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
CUDA_VISIBLE_DEVICES=0 $PY scripts/bl.py selftest --config configs/blself.yaml --out /dev/null 2>&1 | grep -E 'nothing dropped|fastv|visionzip|rlt|codecsight|SELFTEST|Error|Traceback' | cut -c1-160 | tee $L/ntgm-$J-self.log
grep -q SELFTEST_OK $L/ntgm-$J-self.log || { echo "selftest failed"; exit 1; }
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
# The same mechanisms with OUR motion map as the signal (user's request, 2026-11-14), at the rectangle's 1015 tokens.
# RLT and CodecSight with the motion map are one and the same selection (first frame whole + moving cells): one row.
( export CUDA_VISIBLE_DEVICES=0; row fastv1015tgm --mode fastv --budget 1015 --signal tgm ) > $L/ntgm-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; row vz1015tgm --mode visionzip --budget 1015 --signal tgm ) > $L/ntgm-$J-g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; row rlttgm --mode rlt --signal tgm ) > $L/ntgm-$J-g2.log 2>&1 &
wait; grep -h -c '^DONE' $L/ntgm-$J-g*.log; grep -h -E 'FAILED|Traceback|Error' $L/ntgm-$J-g*.log | sort | uniq -c | head
echo NTGM_DONE
