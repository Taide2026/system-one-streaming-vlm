#!/bin/bash
#SBATCH --job-name=s1-soft2
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009,25a-hgpn123
#SBATCH --time=05:00:00
#SBATCH --output=/work/u9117820/s1-logs/soft2-%j.out
# Sharpness at a fixed token count, more models and MSAD (2026-11-16; follows soft.sbatch). The 360p frame is
# area-downscaled to height <via> and enlarged back (input form "soften"; smallvlm.py --via for Qwen3-VL-4B).
#   sbatch soft2.sbatch <cvadall|msad> <zs|f0|flzs|flf1|flf2|qwen> <via> [<via> ...]
set -uo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
ds=$1; m=$2; shift 2
case $ds in cvadall) NT=configs/cvadall_zs_v360.yaml; FT=configs/cvad_fl_tpl_v360.yaml; QT=configs/cvadall_yn_v360.yaml; Q=svcvad ;;
            msad)    NT=configs/msad_tpl_v360.yaml;   FT=configs/msad_fl_tpl_v360.yaml; QT=configs/msad_yn_v360.yaml;    Q=svmsad ;; esac
case $m in zs) A=""; T=$NT ;; f0) A=$F0; T=$NT ;; flzs) A=""; T=$FT ;; flf1|flf2) A=results/nano4/floor_d3flash_c_s${m#flf}/20261107-044624; T=$FT ;; qwen) A=""; T=$QT ;; esac
[ -z "$A" ] || [ -d $A/adapter ] || { echo "missing adapter $A"; exit 1; }
for via in "$@"; do
  if [ $m = qwen ]; then
    out=results/nano4/${Q}_qwen3-vl-4b_yn_v360s$via/run
    [ -f $out/run.json ] || $PY scripts/smallvlm.py --model $W/models-ext/qwen3-vl-4b --config $QT --out $out --factor 1 --via $via 2>&1 | grep --line-buffered -E 'answer tokens|resuming|0000 windows|SMALLVLM_DONE|Error|Traceback'
    $PY scripts/cvad-metrics.py ${Q}_qwen3-vl-4b_yn_v180up2 ${Q}_qwen3-vl-4b_yn_v360s$via ${Q}_qwen3-vl-4b_yn_v360 2>&1 | cut -c1-200 || true
  else
    ev ${ds}_${m}_v360s$via "$A" soft:$via $T 2>&1 | grep -v Loading | cut -c1-300
    $PY scripts/cvad-metrics.py ${ds}_${m}_v180up2 ${ds}_${m}_v360s$via ${ds}_${m}_v360 2>&1 | cut -c1-200 || true
  fi
done
echo SOFT2_DONE
