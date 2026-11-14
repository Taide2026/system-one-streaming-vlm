#!/bin/bash
#SBATCH --job-name=s1-upy3
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:6
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn009
#SBATCH --time=1-12:00:00
#SBATCH --output=/work/u9117820/s1-logs/upy3-%j.out
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs
# Reviewer item 3 (2026-11-11): group A's 1x and all-frame bicubic 2x on the 800 XD-Violence test videos, for three
# models: zero-shot d3-nano, the UCF-trained floor (transfer) and the XD-trained floor (in-domain).
fetch $F0 $FXD
T=configs/ft_xd_v1_d3nano_eval.yaml
CUDA_VISIBLE_DEVICES=0 ev upxd_zs_1x "" frames $T > $L/upy3-g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ev upxd_zs_2x "" up:2 $T > $L/upy3-g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 ev upxd_floorucf_1x $F0 frames $T > $L/upy3-g2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 ev upxd_floorucf_2x $F0 up:2 $T > $L/upy3-g3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 ev upxd_floorxd_1x $FXD frames $T > $L/upy3-g4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 ev upxd_floorxd_2x $FXD up:2 $T > $L/upy3-g5.log 2>&1 &
wait; echo UPY3_DONE
