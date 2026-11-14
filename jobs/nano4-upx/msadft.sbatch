#!/bin/bash
#SBATCH --job-name=s1-msadft
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:3
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/msadft-%j.out
# MSAD in-domain fine-tuning (2026-11-14): d3-nano from the release, Protocol-ii TRAINING videos at 180p, three seeds.
# Recipe = the UCF-Crime one (arm C: 50% V1 soft coverage + 25% class naming + 25% text replay, KL anchor 0.5, LoRA r64,
# lr 1.5e-4) except: the class task names MSAD's own 11 classes + Normal, balanced windows limited by the positives available,
# four epochs (fixed before any test score; MSAD has far fewer windows than UCF-Crime). Held-out: 10% of TRAINING videos.
# Then the same evaluations as msad.sbatch / boxes.sbatch on the 221 test videos.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; ROOT=$W/datasets/msad
nice $PY scripts/msad-train-prep.py $ROOT --jobs 30 | tail -6
mk() {  # $1 seed
  $PY - $1 $ROOT <<'PYEOF'
import sys, yaml
from pathlib import Path
from s1stream.traindata import read_hivau, split_videos, windows_of
seed, root = int(sys.argv[1]), Path(sys.argv[2])
items = read_hivau(root / "prep/msad_train.json", root / "prep", "train180", True, False)
tr, held = split_videos(items, 0.1, seed)
cnt = lambda vs: (sum(r.y > 0 for v in vs for r in windows_of(v, 2.0, 2.0, 8)), sum(r.y == 0 for v in vs for r in windows_of(v, 2.0, 2.0, 8)))
(p, n), (hp, hn) = cnt(tr), cnt(held); k = min(p, n); kv = min(hp, hn)
d = yaml.safe_load(open("configs/fov2_d3nano_c.yaml")); name = f"msadft_d3nano_s{seed}"
d["name"] = name
d["data"] = {"root": str(root / "prep"), "asset": "ucf-crime", "hivau_json": str(root / "prep/msad_train.json"), "videos_dir": "train180", "flat": True,
             "n_val": 2 * kv, "holdout": 0.1, "seed": seed}
CLS = ["Normal", "Assault", "Explosion", "Fighting", "Fire", "Object falling", "People falling", "Robbery", "Shooting", "Traffic accident", "Vandalism", "Water incident"]
d["tasks"] = {"v1": {"n": 2 * k}, "c14": {"n": k, "options": CLS}, "replay": {"n": k, "split": "train", "per_config": 400}}   # 50 / 25 / 25, as on UCF-Crime
d["input_form"] = {"name": "frames"}
d["optim"].update(grad_checkpoint=False, workers=6, batch=8, accumulate=4, epochs=4)
yaml.safe_dump(d, open(f"configs/{name}.yaml", "w"), sort_keys=False)
print(f"{name}: {len(tr)} training videos ({p} positive, {n} negative windows) -> {2*k} video windows + {k} class + {k} replay, 4 epochs; held-out {len(held)} videos, {2*kv} windows", flush=True)
PYEOF
}
done_already() { for d in results/nano4/$1/*/; do [ -f $d/run.json ] && return 0; done 2>/dev/null; return 1; }
b() {  # $1 name, $2 adapter, then bl.py flags (180p base)
  local name=$1 ad=$2; shift 2; done_already $name && return 0
  mkeval $name "$ad" frames configs/msad_tpl_v180.yaml > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) "$@" 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
tok() { $PY -c "import json,glob,sys; d=sorted(x for x in glob.glob('results/nano4/$1/*/') if glob.glob(x+'run.json'))[-1]; t=[json.loads(l)['tokens'] for l in open(d+'scores.jsonl')]; print(round(sum(t)/len(t)))"; }
one() {  # $1 seed
  local s=$1 n=msadft_d3nano_s$1 RUN
  RUN=$(ls -td results/nano4/$n/*/ 2>/dev/null | head -1 || true); RUN=${RUN%/}
  if [ -z "$RUN" ] || [ ! -d $RUN/adapter ]; then
    mk $s; $PY scripts/train-floor.py configs/$n.yaml | grep --line-buffered -E 'step [0-9]*00/|total items|replay|c14|trainable|done|val|Error|Traceback'
    RUN=$(ls -td results/nano4/$n/*/ | head -1); RUN=${RUN%/}; [ -d $RUN/adapter ] || { echo "FAILED train $n"; return 1; }
  fi
  echo "ADAPTER $RUN"; cat $RUN/calibration.json
  local m=in$s
  ev msad_${m}_v180 $RUN frames configs/msad_tpl_v180.yaml
  ev msad_${m}_v180up2 $RUN up:2 configs/msad_tpl_v180.yaml
  ev msad_${m}_v360 $RUN frames configs/msad_tpl_v360.yaml
  b msad_${m}_v180_prect $RUN --mode patch-rect
  b msad_${m}_v180_pboxes $RUN --mode patch-boxes --key 0 7
  b msad_${m}_v180_pdiff $RUN --mode patch-diff --budget $(tok msad_${m}_v180_prect)
  b msad_${m}_v180_ptgm $RUN --mode patch-tgm
}
export -f mk done_already b tok one; export ROOT
for s in 0 1 2; do ( export CUDA_VISIBLE_DEVICES=$s; one $s ) > $L/msadft-$J-g$s.log 2>&1 & done
wait; grep -h -E '^DONE|FAILED|Traceback|Error|ADAPTER|msadft_d3nano' $L/msadft-$J-g*.log | sort | uniq -c | head -40
for s in 0 1 2; do m=in$s; $PY scripts/cvad-metrics.py msad_${m}_v180 msad_${m}_v180up2 msad_${m}_v360 msad_${m}_v180_prect msad_${m}_v180_pboxes msad_${m}_v180_pdiff msad_${m}_v180_ptgm | cut -c1-190; done
echo MSADFT_DONE
