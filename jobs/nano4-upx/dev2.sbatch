#!/bin/bash
#SBATCH --job-name=s1-dev2
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=03:00:00
#SBATCH --output=/work/u9117820/s1-logs/dev2-%j.out
# Why does the rectangle fail on ComplexVAD, and what fixes it? (2026-11-13) DEVELOPMENT SPLIT ONLY: the 30 preview
# videos (list30.txt). The other 83 test videos (list83.txt) are not touched by any layout until one is chosen.
#   crop        the layout as it is, through bl.py (control for the keycrop.py run)
#   patch-rect  the same rectangle, encoded in place (its own position embeddings) instead of as a separate image
#   ctx         the crop plus every other frame whole at the original resolution
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; fetch $F0
grep -q 'videos: list30.txt' configs/cvad_tpl_v180_30.yaml || { echo "template does not pin the 30-video list"; exit 1; }
CUDA_VISIBLE_DEVICES=0 $PY scripts/bl.py selftest --config configs/blself.yaml --out /dev/null 2>&1 | grep -E 'margin|SELFTEST|Error|Traceback' | tail -12
b() {
  local name=cvad30_$1_v180_$3 ad=$2; shift 3
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name "$ad" frames configs/cvad_tpl_v180_30.yaml > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
export -f b
( export CUDA_VISIBLE_DEVICES=0; b zs "" prect --mode patch-rect; b f0 $F0 crop --mode crop ) > $L/dev2-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; b f0 $F0 prect --mode patch-rect ) > $L/dev2-$J-g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; b zs "" ctx --mode ctx; b zs "" crop --mode crop ) > $L/dev2-$J-g2.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; b f0 $F0 ctx --mode ctx ) > $L/dev2-$J-g3.log 2>&1 &
wait; grep -h -E '^DONE|FAILED|Traceback|Error' $L/dev2-$J-g*.log || true
for m in zs f0; do $PY scripts/cvad-metrics.py cvad30_${m}_v180 cvad30_${m}_v180up2 cvad30_${m}_v180kc3 cvad30_${m}_v180_crop cvad30_${m}_v180_prect cvad30_${m}_v180_ctx cvad30_${m}_v180_ptgm cvad30_${m}_v180_pdiff | cut -c1-190; done
echo DEV2_DONE
