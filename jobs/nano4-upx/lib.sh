# Shared by the upy*.sbatch jobs (2026-11-11). Source after cd $REPO.
export W=/work/u9117820 H=/home/u9117820 REPO=/work/u9117820/s1stream
export S1_ASSETS=$W/s1-assets S1_DATASETS=$W/datasets S1_MACHINE=nano4 UCF_CLIPS=$W/datasets/ucf-clips
export HF_HOME=$H/hf-cache UV_CACHE_DIR=$H/uv-cache UV_PYTHON_INSTALL_DIR=$W/uv-python UV_LINK_MODE=copy PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PY=$REPO/.venv/bin/python PYTHONPATH=$REPO/src
export OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;4" OPENCV_FFMPEG_THREADS=4
F0=results/pro6000/floor_d3nano_c/20261031-045123
F1=results/pro6000/floor_d3nano_c_s1/20261102-052314
F2=results/pro6000/floor_d3nano_c_s2/20261102-052314
FXD=results/pro6000/floor_xd_d3nano_c/20261105-002912
fetch() {  # HF adapters by run path
  $PY - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("bear7011/streaming-vad-decision-adapters", allow_patterns=[p + "/*" for p in sys.argv[1:]], local_dir=".")
PY
  for a in "$@"; do [ -d $a/adapter ] || { echo "missing adapter $a"; return 1; }; done
}
form_yaml() {  # $1 form -> the input_form block body (after "input_form:\n")
  case $1 in
    frames|f16) echo "  name: frames" ;;
    mix) printf '  name: foveated\n  frac: 1.0\n  upscale: 2\n' ;;
    fov036) printf '  name: foveated\n  frac: 0.333\n  upscale: 2\n  max_area: 0.036\n' ;;
    up:*) f=${1#up:}; printf '  name: upsample\n  factor: %s\n  kernel: %s\n' "${f%%:*}" "$( [[ $f == *:* ]] && echo ${f#*:} || echo bicubic )" ;;
    *) echo "unknown form $1" >&2; return 1 ;;
  esac
}
mkeval() {  # $1 name, $2 adapter dir or "", $3 form, $4 template (default UCF-290)
  $PY - "$1" "$2" "$3" "${4:-configs/floor290_d3nano_c.yaml}" "$(form_yaml $3)" <<'PY'
import sys, yaml
name, adapter, form, tpl, block = sys.argv[1:]
d = yaml.safe_load(open(tpl)); d["name"] = name
d["input_form"] = yaml.safe_load(block)
if adapter: d["backbone"]["adapter"] = adapter
else: d["backbone"].pop("adapter", None)
if form == "f16": d["window"]["frames"] = 16
yaml.safe_dump(d, open(f"configs/{name}.yaml", "w"), sort_keys=False)
print(name, d["input_form"], d["window"], d["backbone"], flush=True)
PY
}
ev() {  # $1 name, $2 adapter dir or "", $3 form, $4 template
  for d in results/nano4/$1/*/; do [ -f $d/run.json ] && { echo "$1 done already"; return 0; }; done 2>/dev/null
  mkeval "$1" "$2" "$3" "${4:-}"
  $PY -m s1stream.cli run configs/$1.yaml --out results/nano4/$1/$(date +%Y%m%d-%H%M%S) && echo "DONE $1"
}
mktrain() {  # $1 name, $2 seed, $3 form, $4 batch, $5 accumulate
  $PY - "$1" "$2" "$(form_yaml $3)" "$4" "$5" <<'PY'
import sys, yaml
name, seed, block, batch, acc = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
d = yaml.safe_load(open("configs/fov2_d3nano_c.yaml"))
d["name"] = name; d["data"]["seed"] = seed; d["input_form"] = yaml.safe_load(block)
d["optim"]["grad_checkpoint"] = False; d["optim"]["workers"] = 6      # fla + checkpoint -> CheckpointError on nano4
d["optim"]["batch"] = batch; d["optim"]["accumulate"] = acc
yaml.safe_dump(d, open(f"configs/{name}.yaml", "w"), sort_keys=False); print(name, d["input_form"], d["optim"], flush=True)
PY
}
train() {  # $1 name, $2 seed, $3 form  -> sets RUN
  RUN=$(ls -td results/nano4/$1/*/ 2>/dev/null | head -1 || true); RUN=${RUN%/}
  if [ -n "$RUN" ] && [ -d $RUN/adapter ]; then echo "$1 trained already: $RUN"; return 0; fi
  mktrain $1 $2 $3 8 4
  $PY scripts/train-floor.py configs/$1.yaml || { echo "retry $1 at batch 4 x accumulate 8"; sleep 30; mktrain $1 $2 $3 4 8; $PY scripts/train-floor.py configs/$1.yaml; }
  RUN=$(ls -td results/nano4/$1/*/ | head -1); RUN=${RUN%/}; [ -d $RUN/adapter ] || { echo "no adapter for $1"; return 1; }
}
export -f fetch form_yaml mkeval ev mktrain train
export F0 F1 F2 FXD
