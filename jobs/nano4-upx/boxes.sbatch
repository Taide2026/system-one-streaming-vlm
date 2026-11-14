#!/bin/bash
#SBATCH --job-name=s1-boxes
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=04:00:00
#SBATCH --output=/work/u9117820/s1-logs/boxes-%j.out
# One small box per group of moving blocks (bl.py --mode patch-boxes), 2026-11-13 evening. One fixed definition
# (8-connected groups of the moving blocks, each group's bounding box, no margin, encoded in place), run once on:
# MSAD (221 videos), ComplexVAD development 30 and the 83 others, UCF-Crime 290 (three seeds + zero-shot); then latency.
# Note for the write-up: the results of the other layouts on the 83 ComplexVAD videos and on MSAD were known when
# this variant was defined.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; fetch $F0 $F1 $F2
CUDA_VISIBLE_DEVICES=0 $PY scripts/bl.py selftest --config configs/blself.yaml --out /dev/null 2>&1 | grep -E 'patch-boxes|patch-rect|SELFTEST|Error|Traceback' | tail -5
done_already() { for d in results/nano4/$1/*/; do [ -f $d/run.json ] && return 0; done 2>/dev/null; return 1; }
r() {  # $1 name, $2 adapter, $3 template or ""
  local name=$1; done_already $name && return 0
  mkeval $name "$2" frames $3 > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) --mode patch-boxes --key 0 7 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
export -f done_already r
C30=configs/cvad_tpl_v180_30.yaml; C83=configs/cvad_tpl_v180_83.yaml; M=configs/msad_tpl_v180.yaml
( export CUDA_VISIBLE_DEVICES=0; r msad_zs_v180_pboxes "" $M; r cvad83_zs_v180_pboxes "" $C83; r up290_floor_s0_bl_pboxes $F0 "" ) > $L/boxes-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; r msad_f0_v180_pboxes $F0 $M; r cvad83_f0_v180_pboxes $F0 $C83; r up290_floor_s1_bl_pboxes $F1 "" ) > $L/boxes-$J-g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; r cvad30_zs_v180_pboxes "" $C30; r up290_floor_s2_bl_pboxes $F2 "" ) > $L/boxes-$J-g2.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; r cvad30_f0_v180_pboxes $F0 $C30; r up290_zs_bl_pboxes "" "" ) > $L/boxes-$J-g3.log 2>&1 &
wait; grep -h -E '^DONE|FAILED|Traceback|Error' $L/boxes-$J-g*.log || true
for k in ref fla; do CUDA_VISIBLE_DEVICES=0 $PY scripts/bl.py latency --config configs/blself.yaml --out results/nano4/latbl/boxes-$J-$k.json --kernels $k 2>&1 | grep -E '^(reference|fla) +(crop \(ours\)|rect in place|boxes in place|patch tgm|patch diff 1015)|BL_LAT_DONE|Error|Traceback' || true; done
V=$W/datasets/complexvad/list83.txt
for m in zs f0; do
  $PY scripts/cvad-metrics.py msad_${m}_v180up2 msad_${m}_v180_pboxes msad_${m}_v180_prect msad_${m}_v180_pdiff msad_${m}_v180_ptgm | cut -c1-190
  $PY scripts/cvad-metrics.py --videos $V cvadall_${m}_v180up2 cvad83_${m}_v180_pboxes cvad83_${m}_v180_prect cvad83_${m}_v180_pdiff cvad83_${m}_v180_ptgm | cut -c1-190
  $PY scripts/cvad-metrics.py cvad30_${m}_v180up2 cvad30_${m}_v180_pboxes cvad30_${m}_v180_prect cvad30_${m}_v180_pdiff cvad30_${m}_v180_ptgm | cut -c1-190
done
$PY scripts/evalrun.py --events results/pro6000/jev-hard-events.jsonl up290_floor_s0_bl_pboxes up290_floor_s1_bl_pboxes up290_floor_s2_bl_pboxes up290_zs_bl_pboxes | cut -c1-128
echo BOXES_DONE
