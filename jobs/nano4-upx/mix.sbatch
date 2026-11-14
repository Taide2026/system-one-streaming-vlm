#!/bin/bash
#SBATCH --job-name=s1-mix
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/mix-%j.out
# Mixed-resolution visual tokens (2026-11-12), one seed per job (sbatch --export=ALL,SEED=0 mix.sbatch), full UCF-290,
# original-trained floor. 2x frames through the vision tower, then per frame: coarse tokens (2x2 block means)
# everywhere and the fine 2x tokens only in the blocks on the TGM motion region (dilated by one block).
#   mixrep       coarse values replicated: 2x token count, fine content only on the region (information test)
#   mixdrop      one token per coarse block: token count close to 1x, fine tokens only on the region (the method)
#   mixdrop_rand as mixdrop with the same number of fine blocks at random places (does the location matter?)
#   dropcoarse   one token per block, no fine block at all (coarse only, in the same layout)
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs; A=($F0 $F1 $F2); AD=${A[$SEED]}
fetch $AD
pool() {  # $1 mode
  local name=up290_floor_s${SEED}_$1
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name $AD frames > /dev/null
  $PY scripts/mech.py pool --config configs/$name.yaml --mode $1 --kernel bicubic --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) 2>&1 | grep --line-buffered -vE 'Warning|Loading weights|causal_conv1d' && echo "DONE $name"
}
export -f pool; export SEED AD
i=0; for m in mixrep mixdrop mixdrop_rand dropcoarse; do CUDA_VISIBLE_DEVICES=$i pool $m > $L/mix-s${SEED}-$m.log 2>&1 & i=$((i+1)); done
wait; echo MIX_DONE
