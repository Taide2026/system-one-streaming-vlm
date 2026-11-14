#!/bin/bash
#SBATCH --job-name=s1-xd3
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=14:00:00
#SBATCH --output=/work/u9117820/s1-logs/xd3-%j.out
# Three-dataset analysis (2026-11-13), XD-Violence (median 640x360, 800 test videos, 48,408 windows): the two cells
# the pixel-budget account needs. (1) half resolution, whole frame: if XD has pixels to spare, this should cost little.
# (2) key frames + motion crops, for the in-domain fine-tuned model and the zero-shot one.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; fetch $FXD; T=configs/ft_xd_v1_d3nano_eval.yaml
blx() {  # $1 name, $2 adapter, then flags
  local name=$1 ad=$2; shift 2
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name "$ad" frames $T > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
CUDA_VISIBLE_DEVICES=0 ev upxd_floorxd_05x $FXD up:0.5:area $T > $L/xd3-$J-g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ev upxd_zs_05x "" up:0.5:area $T > $L/xd3-$J-g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 blx upxd_floorxd_kc3 $FXD --mode crop --key 0 7 > $L/xd3-$J-g2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 blx upxd_zs_kc3 "" --mode crop --key 0 7 > $L/xd3-$J-g3.log 2>&1 &
wait; grep -h -E '^DONE|FAILED|Traceback|Error' $L/xd3-$J-g*.log || true; echo XD3_DONE
