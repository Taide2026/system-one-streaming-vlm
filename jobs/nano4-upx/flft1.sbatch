#!/bin/bash
#SBATCH --job-name=s1-flft1
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:3
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009,25a-hgpn123
#SBATCH --time=05:00:00
#SBATCH --output=/work/u9117820/s1-logs/flft1-%j.out
# d3-flash fine-tuned on UCF-Crime (seed 1), ComplexVAD 113 test videos, whole frame: 180p | bicubic 2x | real 360p
# (2026-11-15). Zero-shot d3-flash did not change with resolution there (71.6 / 71.3 / 70.3); does the fine-tuned one?
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; A=results/nano4/floor_d3flash_c_s1/20261107-044624; m=flf1
[ -d $A/adapter ] || { echo "missing adapter $A"; exit 1; }
( export CUDA_VISIBLE_DEVICES=0; ev cvadall_${m}_v180 $A frames configs/cvad_fl_tpl_v180.yaml ) > $L/flft1-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; ev cvadall_${m}_v180up2 $A up:2 configs/cvad_fl_tpl_v180.yaml ) > $L/flft1-$J-g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; ev cvadall_${m}_v360 $A frames configs/cvad_fl_tpl_v360.yaml ) > $L/flft1-$J-g2.log 2>&1 &
wait
grep -h -E 'DONE|Error|Traceback' $L/flft1-$J-g*.log | cut -c1-160 || true
$PY scripts/cvad-metrics.py cvadall_${m}_v180 cvadall_${m}_v180up2 cvadall_${m}_v360 2>&1 | cut -c1-200 || true
$PY scripts/cvad-metrics.py cvadall_${m}_v180up2 cvadall_${m}_v360 2>&1 | tail -1 | cut -c1-200 || true
$PY scripts/cvad-metrics.py cvadall_flzs_v180 cvadall_${m}_v180 2>&1 | tail -1 | cut -c1-200 || true
echo FLFT_DONE
