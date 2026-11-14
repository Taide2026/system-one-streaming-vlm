#!/bin/bash
#SBATCH --job-name=s1-cvad
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=04:00:00
#SBATCH --output=/work/u9117820/s1-logs/cvad-%j.out
# ComplexVAD (1920x1080, fixed camera) pattern test (2026-11-13): real resolution against bicubic upsampling against
# key frames + motion crops, zero-shot and fine-tuned (floor seed 0, trained on UCF-Crime only).
#   N=<videos> TAG=<name prefix> TASKS="a b c d e f g" MODELS="zs f0" sbatch cvad.sbatch
# h v180 + real v360 key frames and crops | i v360 + real v720 key frames and crops
# a v180 | b v180 bicubic 2x | c v360 real | d v180 key frames + crops | e v360 bicubic 2x | f v720 real | g v360 key frames + crops
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; ROOT=$W/datasets/complexvad; N=${N:-30}; TAG=${TAG:-cvad$N}; TASKS=${TASKS:-"a b c d e f g h i"}; MODELS=${MODELS:-"zs f0"}
G=$(nvidia-smi -L | wc -l)
fetch $F0
export LIST=${LIST:-}; nice $PY scripts/cvad-prep.py $ROOT --jobs 12
for v in v180 v360 v720; do
  $PY - $v $N $ROOT <<'PYEOF'
import sys, yaml
v, n, root = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = yaml.safe_load(open("configs/floor290_d3nano_c.yaml"))
import os
d["dataset"] = {"name": "complexvad", "path": root, "variant": v}
if os.environ.get("LIST"): d["dataset"]["videos"] = os.environ["LIST"]
else: d["dataset"]["n_videos"] = n
yaml.safe_dump(d, open(f"configs/cvad_tpl_{v}_{n}.yaml", "w"), sort_keys=False)
PYEOF
done
kc() {  # $1 name, $2 adapter, $3 variant, then keycrop flags
  local name=$1 ad=$2 v=$3; shift 3
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name "$ad" frames configs/cvad_tpl_${v}_$N.yaml > /dev/null
  $PY scripts/keycrop.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|KEYCROP_DONE|Error|Traceback' && echo "DONE $name"
}
nat() {  # $1 name, $2 adapter, $3 base variant, $4 variant with twice the resolution: key frames + crops of REAL pixels
  local name=$1
  for d in results/nano4/$name/*/; do [ -f $d/run.json ] && { echo "$name done already"; return 0; }; done 2>/dev/null
  mkeval $name "$2" frames configs/cvad_tpl_${3}_$N.yaml > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) --mode crop --key 0 7 --hi $4 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
one() {  # $1 model, $2 task
  local ad=""; [ $1 = f0 ] && ad=$F0
  local n=${TAG}_$1
  case $2 in
    a) ev ${n}_v180 "$ad" frames configs/cvad_tpl_v180_$N.yaml ;;
    b) ev ${n}_v180up2 "$ad" up:2 configs/cvad_tpl_v180_$N.yaml ;;
    c) ev ${n}_v360 "$ad" frames configs/cvad_tpl_v360_$N.yaml ;;
    d) kc ${n}_v180kc3 "$ad" v180 --key 0 7 --rest touch ;;
    e) ev ${n}_v360up2 "$ad" up:2 configs/cvad_tpl_v360_$N.yaml ;;
    f) ev ${n}_v720 "$ad" frames configs/cvad_tpl_v720_$N.yaml ;;
    g) kc ${n}_v360kc3 "$ad" v360 --key 0 7 --rest touch ;;
    h) nat ${n}_v180nat "$ad" v180 v360 ;;
    i) nat ${n}_v360nat "$ad" v360 v720 ;;
  esac
}
export -f kc nat one; export N TAG ROOT
# heavy tasks first, dealt round-robin over the GPUs
LIST=(); for t in f e i g h c b d a; do for m in $MODELS; do [[ " $TASKS " == *" $t "* ]] && LIST+=("$m $t"); done; done
echo "${#LIST[@]} evaluations on $G GPUs: ${LIST[*]}"
for g in $(seq 0 $((G - 1))); do
  ( i=0; for job in "${LIST[@]}"; do
      if [ $((i % G)) -eq $g ]; then CUDA_VISIBLE_DEVICES=$g one $job || echo "FAILED $job"; fi; i=$((i + 1))
    done ) > $L/cvad-${SLURM_JOB_ID}-g$g.log 2>&1 &
done
wait; grep -h -E '^DONE|FAILED|Traceback|Error' $L/cvad-${SLURM_JOB_ID}-g*.log || true; echo CVAD_DONE
