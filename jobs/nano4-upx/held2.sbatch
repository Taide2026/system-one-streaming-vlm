#!/bin/bash
#SBATCH --job-name=s1-held2
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=03:00:00
#SBATCH --output=/work/u9117820/s1-logs/held2-%j.out
# Other people's patch-level selections on the 83 held-out ComplexVAD videos (2026-11-13), at the rectangle's token
# budget (843 per window), so that "the simple rectangle is on a par with patch-level selection" is tested on the same
# held-out videos as the rectangle. Nothing of ours is changed or re-run.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; fetch $F0
done_already() { for d in results/nano4/$1/*/; do [ -f $d/run.json ] && return 0; done 2>/dev/null; return 1; }
c83() {
  local name=cvad83_$1_v180_$3 ad=$2; shift 3; done_already $name && return 0
  mkeval $name "$ad" frames configs/cvad_tpl_v180_83.yaml > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
export -f done_already c83
CUDA_VISIBLE_DEVICES=0 c83 zs "" pdiff --mode patch-diff --budget 843 > $L/held2-$J-g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 c83 f0 $F0 pdiff --mode patch-diff --budget 843 > $L/held2-$J-g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 c83 zs "" ptgm --mode patch-tgm > $L/held2-$J-g2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 c83 f0 $F0 ptgm --mode patch-tgm > $L/held2-$J-g3.log 2>&1 &
wait; grep -h -E '^DONE|FAILED|Traceback|Error' $L/held2-$J-g*.log || true
for m in zs f0; do $PY scripts/cvad-metrics.py --videos $W/datasets/complexvad/list83.txt cvad83_${m}_v180_prect cvad83_${m}_v180_pdiff cvad83_${m}_v180_ptgm cvad83_${m}_v180_crop | cut -c1-190; done
echo HELD2_DONE
