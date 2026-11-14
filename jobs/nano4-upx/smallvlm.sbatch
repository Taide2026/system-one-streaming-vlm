#!/bin/bash
#SBATCH --job-name=s1-svlm
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:5
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=12:00:00
#SBATCH --output=/work/u9117820/s1-logs/svlm-%j.out
# Five other small VLMs, zero-shot, UCF-Crime 290, whole frame 1x and bicubic 2x (2026-11-14). Processor defaults.
# Probe (svprobe-774529): visual tokens per frame at 1x -> 2x: Qwen3-VL-4B 82 -> 302 (follows the input); Gemma 3 4B 259,
# Gemma 4 E4B 268, SmolVLM2-2.2B 1083, InternVL3.5-2B 3331 (all unchanged by enlarging). Runs resume if resubmitted.
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID
one() {  # $1 model
  for f in 1 2; do
    local out=results/nano4/sv_${1}_x$f/run
    [ -f $out/run.json ] && { echo "DONE sv_${1}_x$f (already)"; continue; }
    $PY scripts/smallvlm.py --model $W/models-ext/$1 --config configs/floor290_d3nano_c.yaml --out $out --factor $f 2>&1 | grep --line-buffered -E 'answer tokens|resuming|[05]000 windows|SMALLVLM_DONE|Error|Traceback' | cut -c1-200 && echo "DONE sv_${1}_x$f"
  done
}
export -f one
i=0; for m in qwen3-vl-4b gemma-3-4b gemma-4-e4b smolvlm2-2.2b internvl3.5-2b; do ( export CUDA_VISIBLE_DEVICES=$i; one $m ) > $L/svlm-$J-$m.log 2>&1 & i=$((i+1)); done
wait; grep -h -E '^DONE|answer tokens|Traceback|Error' $L/svlm-$J-*.log | cut -c1-200
for m in qwen3-vl-4b gemma-3-4b gemma-4-e4b smolvlm2-2.2b internvl3.5-2b; do $PY scripts/cvad-metrics.py sv_${m}_x1 sv_${m}_x2 2>&1 | cut -c1-190; done
echo SVLM_DONE
