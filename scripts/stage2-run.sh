#!/usr/bin/env bash
# One stage-2 run on pro6000 (2026-10-24): train configs/ft_ucf_v1_<model>.yaml, then the sharded 290-video eval with
# configs/ft_ucf_v1_<model>_eval.yaml. Inside a container: scripts/stage2-run.sh <model> <shards>
set -euo pipefail
model="$1"; shards="${2:-4}"
cd "$(dirname "$0")/.."
export PY="${PY:-/opt/venv/bin/python}" S1_DATASETS="${S1_DATASETS:-/data/kidflash7011/datasets}" S1_MACHINE="${S1_MACHINE:-pro6000}" PYTHONPATH=src
scripts/ft.sh train "$model"
d=$(ls -d "results/$S1_MACHINE/ft_ucf_v1_$model"/*/ | tail -1); d=${d%/}
sed "s#ADAPTER_RUN_DIR#$d#" "configs/ft_ucf_v1_${model}_eval.yaml" > "$d/eval-config.yaml"
out="results/$S1_MACHINE/ft_ucf_v1_${model}_eval/$(date +%Y%m%d-%H%M%S)-shards"
scripts/eval-shards.sh "$d/eval-config.yaml" ucf290-manifest.json "$out" "$shards"
echo "STAGE2 $model DONE"
