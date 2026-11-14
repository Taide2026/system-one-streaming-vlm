#!/bin/bash
#SBATCH --job-name=s1-upy1
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn009
#SBATCH --time=16:00:00
#SBATCH --output=/work/u9117820/s1-logs/upy1-%j.out
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs
# Reviewer item 5 (2026-11-11): the pure-2x model. All 8 frames bicubic 2x (2,400 tokens), seeds 0-2, each followed
# by its two test inputs (all-frame 2x; original). GPU 3: the original-trained floor seeds on the all-frame 2x input.
fetch $F0 $F1 $F2
te() { local n=up2x_d3nano_c; [ $1 != 0 ] && n=${n}_s$1; train $n $1 up:2; ev up290_pure2x_s$1_all2x $RUN up:2; ev up290_pure2x_s$1_frames $RUN frames; }
export -f te
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$s te $s > $L/upy1-s$s.log 2>&1 & done
( export CUDA_VISIBLE_DEVICES=3; ev up290_floor_s0_all2x $F0 up:2; ev up290_floor_s1_all2x $F1 up:2; ev up290_floor_s2_all2x $F2 up:2 ) > $L/upy1-g3.log 2>&1 &
wait; echo UPY1_DONE
