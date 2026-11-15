#!/bin/bash
#SBATCH --job-name=s1-flft1g
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009,25a-hgpn123
#SBATCH --time=05:00:00
#SBATCH --output=/work/u9117820/s1-logs/flft1g-%j.out
# One evaluation of flft2.sbatch per job, one GPU each (2026-11-15): the account's GPU limit was full, and one-GPU jobs
# start as single GPUs come free.   sbatch flft1g.sbatch <seed> <v180|v180up2|v360>
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
s=$1; v=$2; A=results/nano4/floor_d3flash_c_s$s/20261107-044624; m=flf$s
[ -d $A/adapter ] || { echo "missing adapter $A"; exit 1; }
case $v in v180) f=frames; t=v180 ;; v180up2) f=up:2; t=v180 ;; v360) f=frames; t=v360 ;; esac
ev cvadall_${m}_$v $A $f configs/cvad_fl_tpl_$t.yaml 2>&1 | grep -v Loading | cut -c1-300
$PY scripts/cvad-metrics.py cvadall_${m}_v180 cvadall_${m}_v180up2 cvadall_${m}_v360 2>&1 | cut -c1-200 || true
echo FLFT1G_DONE
