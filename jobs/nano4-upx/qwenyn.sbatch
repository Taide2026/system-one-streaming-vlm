#!/bin/bash
#SBATCH --job-name=s1-qwenyn
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009,25a-hgpn123
#SBATCH --time=06:00:00
#SBATCH --output=/work/u9117820/s1-logs/qwenyn-%j.out
# Qwen3-VL-4B, zero-shot, neutral question, ComplexVAD: the other two inputs (2026-11-15), after 180p gave Macro 69.5
# with this question (39.1 with the crime-list prompt): 180p bicubic 2x, and real 360p.
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; M=$W/models-ext/qwen3-vl-4b
$PY - <<'PYEOF'
import yaml
d = yaml.safe_load(open("configs/cvadall_zs_v360.yaml")); d["question"] = {"prompt": "yes_no"}; d["name"] = "cvadall_yn_v360"
yaml.safe_dump(d, open("configs/cvadall_yn_v360.yaml", "w"), sort_keys=False)
PYEOF
one() {  # $1 run name, $2 config, $3 factor
  local out=results/nano4/$1/run; [ -f $out/run.json ] && return 0
  $PY scripts/smallvlm.py --model $M --config $2 --out $out --factor $3 2>&1 | grep --line-buffered -E 'answer tokens|resuming|[05]000 windows|SMALLVLM_DONE|Error|Traceback' && echo "DONE $1"
}
( export CUDA_VISIBLE_DEVICES=0; one svcvad_qwen3-vl-4b_yn_v180up2 configs/cvadall_yn_v180.yaml 2 ) > $L/qwenyn-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; one svcvad_qwen3-vl-4b_yn_v360 configs/cvadall_yn_v360.yaml 1 ) > $L/qwenyn-$J-g1.log 2>&1 &
wait
cat $L/qwenyn-$J-g*.log | cut -c1-200
$PY scripts/cvad-metrics.py svcvad_qwen3-vl-4b_yn_v180 svcvad_qwen3-vl-4b_yn_v180up2 svcvad_qwen3-vl-4b_yn_v360 2>&1 | cut -c1-200 || true
$PY scripts/cvad-metrics.py svcvad_qwen3-vl-4b_yn_v180up2 svcvad_qwen3-vl-4b_yn_v360 2>&1 | tail -1 | cut -c1-200 || true
echo QWENYN_DONE
