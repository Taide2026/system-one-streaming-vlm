#!/bin/bash
#SBATCH --job-name=s1-msadflft
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009,25a-hgpn123
#SBATCH --time=03:00:00
#SBATCH --output=/work/u9117820/s1-logs/msadflft-%j.out
# d3-flash fine-tuned on UCF-Crime (seed $1), MSAD test videos, whole frame: 180p | bicubic 2x | real 360p (2026-11-15).
# Needs configs/msad_fl_tpl_v180.yaml / _v360.yaml written by msad2.sbatch.   sbatch msadflft.sbatch <seed>
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
s=$1; A=results/nano4/floor_d3flash_c_s$s/20261107-044624; m=flf$s
[ -d $A/adapter ] || { echo "missing adapter $A"; exit 1; }
{ ev msad_${m}_v180 $A frames configs/msad_fl_tpl_v180.yaml; ev msad_${m}_v180up2 $A up:2 configs/msad_fl_tpl_v180.yaml; ev msad_${m}_v360 $A frames configs/msad_fl_tpl_v360.yaml; } 2>&1 | grep -v Loading | cut -c1-300
$PY scripts/cvad-metrics.py msad_${m}_v180 msad_${m}_v180up2 msad_${m}_v360 2>&1 | cut -c1-200 || true
$PY scripts/cvad-metrics.py msad_${m}_v180up2 msad_${m}_v360 2>&1 | tail -1 | cut -c1-200 || true
echo MSADFLFT_DONE
