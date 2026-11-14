#!/bin/bash
#SBATCH --job-name=s1-msad
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=04:00:00
#SBATCH --output=/work/u9117820/s1-logs/msad-%j.out
# MSAD, Protocol-ii test split (120 abnormal + 120 normal), no training (2026-11-13): the third dataset.
# High-resolution sources, anomalies that move. Zero-shot d3-nano and the UCF-Crime fine-tune (seed 0), both transfer.
#   whole frame: 180p | 180p bicubic 2x | real 360p | 360p bicubic 2x | real 720p
#   180p base, 2x layout: rectangle in place (ours), most-changed patches at the rectangle's token count, moving patches
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; ROOT=$W/datasets/msad; fetch $F0
nice $PY scripts/msad-prep.py $ROOT --jobs 24 | tail -8
for v in v180 v360 v720; do
  $PY - $v $ROOT <<'PYEOF'
import sys, yaml
v, root = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open("configs/floor290_d3nano_c.yaml"))
d["dataset"] = {"name": "msad", "path": root, "variant": v, "videos_from": "v720"}     # the same videos in every variant
yaml.safe_dump(d, open(f"configs/msad_tpl_{v}.yaml", "w"), sort_keys=False)
PYEOF
done
done_already() { for d in results/nano4/$1/*/; do [ -f $d/run.json ] && return 0; done 2>/dev/null; return 1; }
b() {  # $1 name, $2 adapter, then bl.py flags (180p base)
  local name=$1 ad=$2; shift 2; done_already $name && return 0
  mkeval $name "$ad" frames configs/msad_tpl_v180.yaml > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
tok() { $PY -c "import json,glob,sys; d=sorted(x for x in glob.glob('results/nano4/$1/*/') if glob.glob(x+'run.json'))[-1]; t=[json.loads(l)['tokens'] for l in open(d+'scores.jsonl')]; print(round(sum(t)/len(t)))"; }
run() {  # $1 model tag, $2 adapter
  local m=$1 ad=$2
  ev msad_${m}_v180 "$ad" frames configs/msad_tpl_v180.yaml
  ev msad_${m}_v180up2 "$ad" up:2 configs/msad_tpl_v180.yaml
  ev msad_${m}_v360 "$ad" frames configs/msad_tpl_v360.yaml
  b msad_${m}_v180_prect "$ad" --mode patch-rect
  b msad_${m}_v180_pdiff "$ad" --mode patch-diff --budget $(tok msad_${m}_v180_prect)
  b msad_${m}_v180_ptgm "$ad" --mode patch-tgm
}
heavy() { ev msad_$1_v360up2 "$2" up:2 configs/msad_tpl_v360.yaml; ev msad_$1_v720 "$2" frames configs/msad_tpl_v720.yaml; }
export -f done_already b tok run heavy
( export CUDA_VISIBLE_DEVICES=0; run zs "" ) > $L/msad-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; run f0 $F0 ) > $L/msad-$J-g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; heavy zs "" ) > $L/msad-$J-g2.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; heavy f0 $F0 ) > $L/msad-$J-g3.log 2>&1 &
wait; grep -h -E '^DONE|FAILED|Traceback|Error' $L/msad-$J-g*.log | sort | uniq -c | head -30
for m in zs f0; do $PY scripts/cvad-metrics.py msad_${m}_v180 msad_${m}_v180up2 msad_${m}_v360 msad_${m}_v360up2 msad_${m}_v720 msad_${m}_v180_prect msad_${m}_v180_pdiff msad_${m}_v180_ptgm | cut -c1-190; done
echo MSAD_DONE
