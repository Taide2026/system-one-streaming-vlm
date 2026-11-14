#!/bin/bash
#SBATCH --job-name=s1-final2
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=05:00:00
#SBATCH --output=/work/u9117820/s1-logs/final2-%j.out
# The chosen fix, run once on data it was not chosen on (2026-11-13). Chosen on the 30 development videos of ComplexVAD:
# the rectangle encoded in place (bl.py --mode patch-rect) instead of as a separate image.
#   ComplexVAD, the 83 held-out test videos (list83.txt): rectangle in place, and the old separate-image crop for contrast.
#   UCF-Crime 290, three fine-tuned seeds and zero-shot: rectangle in place.
#   Then latency of every layout on one GPU, both kernels.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; ROOT=$W/datasets/complexvad; fetch $F0 $F1 $F2
$PY - $ROOT <<'PYEOF'
import sys, yaml
d = yaml.safe_load(open("configs/floor290_d3nano_c.yaml"))
d["dataset"] = {"name": "complexvad", "path": sys.argv[1], "variant": "v180", "videos": "list83.txt"}
yaml.safe_dump(d, open("configs/cvad_tpl_v180_83.yaml", "w"), sort_keys=False)
PYEOF
done_already() { for d in results/nano4/$1/*/; do [ -f $d/run.json ] && return 0; done 2>/dev/null; return 1; }
c83() {  # $1 tag, $2 adapter, $3 name, mode flags
  local name=cvad83_$1_v180_$3 ad=$2; shift 3; done_already $name && return 0
  mkeval $name "$ad" frames configs/cvad_tpl_v180_83.yaml > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
ucf() {  # $1 tag, $2 adapter
  local name=up290_$1_bl_prect; done_already $name && return 0
  mkeval $name "$2" frames > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) --mode patch-rect 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
export -f done_already c83 ucf
( export CUDA_VISIBLE_DEVICES=0; c83 zs "" prect --mode patch-rect; ucf floor_s0 $F0 ) > $L/final2-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; c83 f0 $F0 prect --mode patch-rect; ucf floor_s1 $F1 ) > $L/final2-$J-g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; c83 zs "" crop --mode crop; ucf floor_s2 $F2 ) > $L/final2-$J-g2.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; c83 f0 $F0 crop --mode crop; ucf zs "" ) > $L/final2-$J-g3.log 2>&1 &
wait; grep -h -E '^DONE|FAILED|Traceback|Error' $L/final2-$J-g*.log || true
for k in ref fla; do
  CUDA_VISIBLE_DEVICES=0 $PY scripts/bl.py latency --config configs/blself.yaml --out results/nano4/latbl/$J-$k.json --kernels $k 2>&1 | grep -E '^(reference|fla) |BL_LAT_DONE|Error|Traceback' || true
done
for m in zs f0; do $PY scripts/cvad-metrics.py --videos $ROOT/list83.txt cvadall_${m}_v180 cvadall_${m}_v180up2 cvadall_${m}_v360 cvad83_${m}_v180_prect cvad83_${m}_v180_crop | cut -c1-190; done
$PY scripts/evalrun.py --events results/pro6000/jev-hard-events.jsonl up290_floor_s0_bl_prect up290_floor_s1_bl_prect up290_floor_s2_bl_prect up290_zs_bl_prect | cut -c1-128
echo FINAL2_DONE
