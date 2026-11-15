#!/bin/bash
#SBATCH --job-name=s1-msad2
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009,25a-hgpn123
#SBATCH --time=06:00:00
#SBATCH --output=/work/u9117820/s1-logs/msad2-%j.out
# Interpolated against real pixels on a second dataset for the two models that did not follow d3-nano on ComplexVAD
# (2026-11-15): MSAD test videos, zero-shot, whole frame, 180p | 180p bicubic 2x | real 360p.
#   GPU0  d3-flash (8.4B)                      -> msad_flzs_v180 | _v180up2 | _v360
#   GPU1  Qwen3-VL-4B, neutral yes/no question -> svmsad_qwen3-vl-4b_yn_v180 | _v180up2 | _v360
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; M=$W/models-ext/qwen3-vl-4b
$PY - <<'PYEOF'
import yaml
fl = yaml.safe_load(open("configs/floor290_d3flash_c.yaml"))["backbone"]; fl.pop("adapter", None)
for v in ("v180", "v360"):
    d = yaml.safe_load(open(f"configs/msad_tpl_{v}.yaml")); d["backbone"] = dict(fl); d["name"] = f"msad_fl_tpl_{v}"
    yaml.safe_dump(d, open(f"configs/msad_fl_tpl_{v}.yaml", "w"), sort_keys=False)
    d = yaml.safe_load(open(f"configs/msad_tpl_{v}.yaml")); d["question"] = {"prompt": "yes_no"}; d["name"] = f"msad_yn_{v}"
    yaml.safe_dump(d, open(f"configs/msad_yn_{v}.yaml", "w"), sort_keys=False)
print("d3-flash backbone:", fl, flush=True)
PYEOF
one() {  # $1 run name, $2 config, $3 factor
  local out=results/nano4/$1/run; [ -f $out/run.json ] && return 0
  $PY scripts/smallvlm.py --model $M --config $2 --out $out --factor $3 2>&1 | grep --line-buffered -E 'answer tokens|resuming|[05]000 windows|SMALLVLM_DONE|Error|Traceback' && echo "DONE $1"
}
( export CUDA_VISIBLE_DEVICES=0; ev msad_flzs_v180 "" frames configs/msad_fl_tpl_v180.yaml; ev msad_flzs_v180up2 "" up:2 configs/msad_fl_tpl_v180.yaml; ev msad_flzs_v360 "" frames configs/msad_fl_tpl_v360.yaml ) > $L/msad2-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; one svmsad_qwen3-vl-4b_yn_v180 configs/msad_yn_v180.yaml 1; one svmsad_qwen3-vl-4b_yn_v180up2 configs/msad_yn_v180.yaml 2; one svmsad_qwen3-vl-4b_yn_v360 configs/msad_yn_v360.yaml 1 ) > $L/msad2-$J-g1.log 2>&1 &
wait
grep -h -E 'DONE|Error|Traceback|answer' $L/msad2-$J-g*.log | cut -c1-160 || true
$PY scripts/cvad-metrics.py msad_zs_v180 msad_zs_v180up2 msad_zs_v360 2>&1 | cut -c1-200 || true
$PY scripts/cvad-metrics.py msad_flzs_v180 msad_flzs_v180up2 msad_flzs_v360 2>&1 | cut -c1-200 || true
$PY scripts/cvad-metrics.py msad_flzs_v180up2 msad_flzs_v360 2>&1 | tail -1 | cut -c1-200 || true
$PY scripts/cvad-metrics.py svmsad_qwen3-vl-4b_yn_v180 svmsad_qwen3-vl-4b_yn_v180up2 svmsad_qwen3-vl-4b_yn_v360 2>&1 | cut -c1-200 || true
$PY scripts/cvad-metrics.py svmsad_qwen3-vl-4b_yn_v180up2 svmsad_qwen3-vl-4b_yn_v360 2>&1 | tail -1 | cut -c1-200 || true
echo MSAD2_DONE
