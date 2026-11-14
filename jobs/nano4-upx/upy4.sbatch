#!/bin/bash
#SBATCH --job-name=s1-upy4
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn009
#SBATCH --time=06:00:00
#SBATCH --output=/work/u9117820/s1-logs/upy4-%j.out
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs
# Reviewer item 4 (2026-11-11): cost per input. The first 2 windows of each of the 290 test videos (5 warm-up windows
# left out), one GPU, nothing else on it: zero-shot and floor, each input, with the reference kernels (fla hidden:
# the numbers latency claims may use) and with the fast kernels (what the accuracy runs used). Decode and the
# upsampling itself are inside the timed block.
fetch $F0
lat() {  # $1 model tag, $2 adapter or "", $3 form, $4 kernels (ref|fla)
  local tag=${3//:/-}; local name=lat_${1}_${tag}; local out=results/nano4/latency/$4/$name
  [ -f $out/latency.json ] && return 0; rm -rf $out; mkdir -p $(dirname $out)
  mkeval $name "$2" $3 > /dev/null
  if [ $4 = ref ]; then
    $PY -c "import sys, runpy; sys.modules['fla'] = None; sys.argv = ['latency-sample.py', 'configs/$name.yaml', '--out', '$out']; runpy.run_path('scripts/latency-sample.py', run_name='__main__')" > $L/upy4-$4-$name.log 2>&1 || echo "FAILED $4 $name"
  else
    $PY scripts/latency-sample.py configs/$name.yaml --out $out > $L/upy4-$4-$name.log 2>&1 || echo "FAILED $4 $name"
  fi
  [ -f $out/latency.json ] && $PY -c "import json; d=json.load(open('$out/latency.json')); print('$4', '$name', 'p50', round(d['p50_ms']), 'p95', round(d['p95_ms']), 'rtf', round(d['rtf_p95'],3), d['kernels'])" || true
}
for k in ref fla; do for form in frames up:1.25 up:1.5 up:1.75 up:2 up:3 mix f16 fov036; do
  lat zs "" $form $k; lat floor $F0 $form $k
done; done
echo UPY4_DONE
