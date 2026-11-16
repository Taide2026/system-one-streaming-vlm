#!/bin/bash
#SBATCH --job-name=s1-soft
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009,25a-hgpn123
#SBATCH --time=05:00:00
#SBATCH --output=/work/u9117820/s1-logs/soft-%j.out
# Sharpness at a fixed token count (2026-11-16): ComplexVAD 360p frames area-downscaled to height $2 and enlarged back
# (input form "soften"). Between "180p enlarged 2x" (via 180) and real 360p. Does the fine-tuned d3-flash score fall
# as the frame gets sharper while the zero-shot d3-nano rises?   sbatch soft.sbatch <flf1|zs> <via>
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
m=$1; via=$2
case $m in flf1) A=results/nano4/floor_d3flash_c_s1/20261107-044624; T=configs/cvad_fl_tpl_v360.yaml ;; zs) A=""; T=configs/cvadall_zs_v360.yaml ;; esac
ev cvadall_${m}_v360s$via "$A" soft:$via $T 2>&1 | grep -v Loading | cut -c1-300
$PY scripts/cvad-metrics.py cvadall_${m}_v180up2 cvadall_${m}_v360s$via cvadall_${m}_v360 2>&1 | cut -c1-200 || true
echo SOFT_DONE
