#!/bin/bash
#SBATCH --job-name=s1-final3
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/final3-%j.out
# Completing the experiments of the chosen layout, the rectangle encoded in place (2026-11-13), full UCF-290:
#   three key frames (0, 3, 7), three fine-tuned seeds of d3-nano;
#   the second model, d3-flash: zero-shot and its two fine-tuned seeds, two key frames.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; fetch $F0 $F1 $F2; TF=configs/floor290_d3flash_c_s1.yaml
S1=results/nano4/floor_d3flash_c_s1/20261107-044624; S2=results/nano4/floor_d3flash_c_s2/20261107-044624
done_already() { for d in results/nano4/$1/*/; do [ -f $d/run.json ] && return 0; done 2>/dev/null; return 1; }
r() {  # $1 name, $2 adapter, $3 template or "", then flags
  local name=$1 ad=$2 tpl=$3; shift 3; done_already $name && return 0
  mkeval $name "$ad" frames $tpl > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) --mode patch-rect "$@" 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
export -f done_already r
( export CUDA_VISIBLE_DEVICES=0; r fl290_s1_prect $S1 $TF --key 0 7; r up290_floor_s0_bl_prect3 $F0 "" --key 0 3 7 ) > $L/final3-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; r fl290_s2_prect $S2 $TF --key 0 7; r up290_floor_s1_bl_prect3 $F1 "" --key 0 3 7 ) > $L/final3-$J-g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; r fl290_zs_prect "" $TF --key 0 7; r up290_floor_s2_bl_prect3 $F2 "" --key 0 3 7 ) > $L/final3-$J-g2.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; mkeval lat_flash_zs "" frames $TF > /dev/null
  for k in ref fla; do $PY scripts/bl.py latency --config configs/lat_flash_zs.yaml --out results/nano4/latbl/flash-$J-$k.json --kernels $k 2>&1 | grep -E '^(reference|fla) |BL_LAT_DONE|Error|Traceback' || true; done ) > $L/final3-$J-g3.log 2>&1 &
wait; grep -h -E '^DONE|FAILED|Traceback|Error' $L/final3-$J-g*.log || true; cat $L/final3-$J-g3.log | cut -c1-200
$PY scripts/evalrun.py --events results/pro6000/jev-hard-events.jsonl up290_floor_s0_bl_prect3 up290_floor_s1_bl_prect3 up290_floor_s2_bl_prect3 fl290_zs_prect fl290_s1_prect fl290_s2_prect | cut -c1-128
echo FINAL3_DONE
