#!/bin/bash
#SBATCH --job-name=s1-second2
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=04:00:00
#SBATCH --output=/work/u9117820/s1-logs/second2-%j.out
# The second halves of second.sbatch (d3-flash zero-shot ComplexVAD 180p and real 360p), started at once on two more
# GPUs (2026-11-15, user asked to speed up). A placeholder run dir "00-elsewhere/run.json" makes job 780356 skip them;
# it is removed here once the real run has written its own run.json.
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID
one() {  # $1 name, $2 template
  mkdir -p results/nano4/$1/00-elsewhere; echo '{"placeholder": "running in second2"}' > results/nano4/$1/00-elsewhere/run.json
  mkeval "$1" "" frames "$2"
  $PY -m s1stream.cli run configs/$1.yaml --out results/nano4/$1/$(date +%Y%m%d-%H%M%S) && echo "DONE $1"
  rm -rf results/nano4/$1/00-elsewhere
}
( export CUDA_VISIBLE_DEVICES=0; one cvadall_flzs_v180 configs/cvad_fl_tpl_v180.yaml ) > $L/second2-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; one cvadall_flzs_v360 configs/cvad_fl_tpl_v360.yaml ) > $L/second2-$J-g1.log 2>&1 &
wait
grep -h -E 'DONE|Error|Traceback' $L/second2-$J-g*.log | cut -c1-160 || true
$PY scripts/cvad-metrics.py cvadall_flzs_v180 cvadall_flzs_v180up2 cvadall_flzs_v360 2>&1 | cut -c1-200 || true
echo SECOND2_DONE
