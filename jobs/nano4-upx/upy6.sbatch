#!/bin/bash
#SBATCH --job-name=s1-upy6
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn009
#SBATCH --time=16:00:00
#SBATCH --output=/work/u9117820/s1-logs/upy6-%j.out
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs
# Reviewer items 6 and 2 (2026-11-11). Item 6: the crop model with the gate from the TRAINING split (hot-area upper
# tercile 3.6%, not the test-derived 4.5%), seeds 0-2, tested with the same gate. Item 2: the equal-token baseline,
# 16 original frames per window (1,280 tokens), at test time on the three floor seeds and zero-shot.
fetch $F0 $F1 $F2
te() { local n=fov2g036_d3nano_c; [ $1 != 0 ] && n=${n}_s$1; train $n $1 fov036; ev up290_fov2g036_s$1 $RUN fov036; }
export -f te
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$s te $s > $L/upy6-s$s.log 2>&1 & done
( export CUDA_VISIBLE_DEVICES=3; ev up290_zs_f16 "" f16; ev up290_floor_s2_f16 $F2 f16; ev up290_floor_s1_f16 $F1 f16; ev up290_floor_s0_f16 $F0 f16 ) > $L/upy6-g3.log 2>&1 &
wait; echo UPY6_DONE
