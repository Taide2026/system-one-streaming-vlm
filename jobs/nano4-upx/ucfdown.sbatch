#!/bin/bash
#SBATCH --job-name=s1-ucfdown
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=03:00:00
#SBATCH --output=/work/u9117820/s1-logs/ucfdown-%j.out
# Three-dataset analysis (2026-11-13), UCF-Crime below its native resolution: whole frame at 0.5x and 0.75x (area
# downscale), zero-shot and three fine-tuned seeds. The low end of the pixels-per-event curve.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; fetch $F0 $F1 $F2
( for f in 05 075; do x=0.${f#0}; CUDA_VISIBLE_DEVICES=0 ev up290_zs_dn$f "" up:$x:area; CUDA_VISIBLE_DEVICES=0 ev up290_floor_s0_dn$f $F0 up:$x:area; done ) > $L/ucfdown-$J-g0.log 2>&1 &
( for f in 05 075; do x=0.${f#0}; CUDA_VISIBLE_DEVICES=1 ev up290_floor_s1_dn$f $F1 up:$x:area; CUDA_VISIBLE_DEVICES=1 ev up290_floor_s2_dn$f $F2 up:$x:area; done ) > $L/ucfdown-$J-g1.log 2>&1 &
wait; grep -h -E '^DONE|FAILED|Traceback|Error' $L/ucfdown-$J-g*.log || true; echo UCFDOWN_DONE
