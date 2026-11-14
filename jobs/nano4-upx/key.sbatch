#!/bin/bash
#SBATCH --job-name=s1-key
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:6
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/key-%j.out
# Keyframe + motion tokens (2026-11-12), effect only, full UCF-290. sbatch --export=ALL,MODEL=zs|floor_s0 key.sbatch
#   k1  one fine key frame (frame 3) + touch blocks of the other 7 frames, nothing else       ~520 tokens
#   k2  one fine key frame + loose blocks of the other frames                                   ~1000
#   k3  two fine key frames (0 and 7) + touch blocks of the other 6                             ~790
#   k4  one COARSE key frame + touch blocks of the other frames                                 ~330
#   k5  no key frame: touch blocks only, all 8 frames (control: is the background needed)       ~250
#   k6  one fine key frame + touch blocks + coarse tokens in the other frames (control)         ~1020
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs; AD=""; [ "$MODEL" = floor_s0 ] && { fetch $F0; AD=$F0; }
run() {  # $1 tag, then keyframe.py flags
  local name=up290_${MODEL}_$1; shift
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name "$AD" frames > /dev/null
  $PY scripts/keyframe.py --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|KEY_DONE|Error|Traceback' && echo "DONE $name"
}
export -f run; export MODEL AD
CUDA_VISIBLE_DEVICES=0 run k1 --key 3 --key-layout fine --rest touch > $L/key-$MODEL-k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 run k2 --key 3 --key-layout fine --rest loose > $L/key-$MODEL-k2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 run k3 --key 0 7 --key-layout fine --rest touch > $L/key-$MODEL-k3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 run k4 --key 3 --key-layout coarse --rest touch > $L/key-$MODEL-k4.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 run k5 --key --rest touch > $L/key-$MODEL-k5.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 run k6 --key 3 --key-layout fine --rest touch --rest-coarse 1 > $L/key-$MODEL-k6.log 2>&1 &
wait; echo KEY_JOB_DONE
