#!/bin/bash
#SBATCH --job-name=s1-svcvad
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:3
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/svcvad-%j.out
# Interpolated against real pixels on a second model (2026-11-14): Qwen3-VL-4B, zero-shot, ComplexVAD 113 test videos,
# whole frame: 180p | 180p bicubic 2x | real 360p. Same windows and prompt as the d3-nano rows of the ladder figure.
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; M=$W/models-ext/qwen3-vl-4b
one() {  # $1 run name, $2 config, $3 factor
  local out=results/nano4/$1/run
  [ -f $out/run.json ] && { echo "DONE $1 (already)"; return 0; }
  $PY scripts/smallvlm.py --model $M --config $2 --out $out --factor $3 2>&1 | grep --line-buffered -E 'answer tokens|resuming|[05]000 windows|SMALLVLM_DONE|Error|Traceback' && echo "DONE $1"
}
export -f one; export M
( export CUDA_VISIBLE_DEVICES=0; one svcvad_qwen3-vl-4b_v180 configs/cvadall_zs_v180.yaml 1 ) > $L/svcvad-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; one svcvad_qwen3-vl-4b_v180up2 configs/cvadall_zs_v180.yaml 2 ) > $L/svcvad-$J-g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; one svcvad_qwen3-vl-4b_v360 configs/cvadall_zs_v360.yaml 1 ) > $L/svcvad-$J-g2.log 2>&1 &
wait
cat $L/svcvad-$J-g*.log | grep -E 'DONE|answer|Error|Traceback' | cut -c1-160 || true
$PY scripts/cvad-metrics.py svcvad_qwen3-vl-4b_v180 svcvad_qwen3-vl-4b_v180up2 svcvad_qwen3-vl-4b_v360 2>&1 | cut -c1-200 || true
$PY scripts/cvad-metrics.py svcvad_qwen3-vl-4b_v180up2 svcvad_qwen3-vl-4b_v360 2>&1 | tail -1 | cut -c1-200 || true
echo SVCVAD_DONE
