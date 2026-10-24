#!/usr/bin/env bash
# XD-Violence train + eval on pro6000 (2026-10-24): train configs/ft_xd_v1_<model>.yaml, then the sharded 800-video eval
# with configs/ft_xd_v1_<model>_eval.yaml (xd800-manifest.json). Inside a container: scripts/stage2-xd-run.sh <model> <shards>
set -euo pipefail
model="$1"; shards="${2:-4}"
cd "$(dirname "$0")/.."
export PY="${PY:-/opt/venv/bin/python}" S1_DATASETS="${S1_DATASETS:-/data/kidflash7011/datasets}" S1_MACHINE="${S1_MACHINE:-pro6000}" PYTHONPATH=src
out_train="results/$S1_MACHINE/ft_xd_v1_$model/$(date +%Y%m%d-%H%M%S)"
$PY -m s1stream.cli train "configs/ft_xd_v1_$model.yaml" --out "$out_train"
sed "s#ADAPTER_RUN_DIR#$out_train#" "configs/ft_xd_v1_${model}_eval.yaml" > "$out_train/eval-config.yaml"
out="results/$S1_MACHINE/ft_xd_v1_${model}_eval/$(date +%Y%m%d-%H%M%S)-shards"
scripts/eval-shards.sh "$out_train/eval-config.yaml" xd800-manifest.json "$out" "$shards"
echo "STAGE2-XD $model DONE"
