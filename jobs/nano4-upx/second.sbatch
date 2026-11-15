#!/bin/bash
#SBATCH --job-name=s1-second
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/second-%j.out
# A second model for "interpolated against real pixels" (2026-11-15), ComplexVAD 113 test videos, whole frame.
#   d3-flash (8.4B, zero-shot): 180p | 180p bicubic 2x | real 360p      -> same family as d3-nano, another size
#   Qwen3-VL-4B, zero-shot, 180p, with the neutral question ("Is something abnormal happening?") instead of the
#   crime-list prompt: it was below chance on ComplexVAD with the latter (svcvad-777339); one run to see whether the
#   prompt is the reason, before spending on the other two inputs.
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID
$PY - <<'PYEOF'
import yaml
fl = yaml.safe_load(open("configs/floor290_d3flash_c.yaml"))["backbone"]; fl.pop("adapter", None)
for v in ("v180", "v360"):
    d = yaml.safe_load(open(f"configs/cvadall_zs_{v}.yaml")); d["backbone"] = dict(fl); d["name"] = f"cvad_fl_tpl_{v}"
    yaml.safe_dump(d, open(f"configs/cvad_fl_tpl_{v}.yaml", "w"), sort_keys=False)
d = yaml.safe_load(open("configs/cvadall_zs_v180.yaml")); d["question"] = {"prompt": "yes_no"}; d["name"] = "cvadall_yn_v180"
yaml.safe_dump(d, open("configs/cvadall_yn_v180.yaml", "w"), sort_keys=False)
print("d3-flash backbone:", fl, flush=True)
PYEOF
qwen() { local out=results/nano4/svcvad_qwen3-vl-4b_yn_v180/run; [ -f $out/run.json ] && return 0
  $PY scripts/smallvlm.py --model $W/models-ext/qwen3-vl-4b --config configs/cvadall_yn_v180.yaml --out $out --factor 1 2>&1 | grep --line-buffered -E 'answer tokens|resuming|0000 windows|SMALLVLM_DONE|Error|Traceback'; }
export -f qwen
# two GPUs, two runs each, in order (2026-11-15: the account's GPU limit was full); the Qwen3-VL-4B trial goes first
( export CUDA_VISIBLE_DEVICES=0; qwen; $PY scripts/cvad-metrics.py svcvad_qwen3-vl-4b_v180 svcvad_qwen3-vl-4b_yn_v180 2>&1 | cut -c1-200; ev cvadall_flzs_v180 "" frames configs/cvad_fl_tpl_v180.yaml ) > $L/second-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; ev cvadall_flzs_v180up2 "" up:2 configs/cvad_fl_tpl_v180.yaml; ev cvadall_flzs_v360 "" frames configs/cvad_fl_tpl_v360.yaml ) > $L/second-$J-g1.log 2>&1 &
wait
grep -h -E 'DONE|Error|Traceback|answer' $L/second-$J-g*.log | cut -c1-160 || true
$PY scripts/cvad-metrics.py cvadall_flzs_v180 cvadall_flzs_v180up2 cvadall_flzs_v360 2>&1 | cut -c1-200 || true
$PY scripts/cvad-metrics.py cvadall_flzs_v180up2 cvadall_flzs_v360 2>&1 | tail -1 | cut -c1-200 || true
$PY scripts/cvad-metrics.py svcvad_qwen3-vl-4b_v180 svcvad_qwen3-vl-4b_yn_v180 2>&1 | cut -c1-200 || true
echo SECOND_DONE
