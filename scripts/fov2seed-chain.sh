#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export S1_DATASETS=${S1_DATASETS:-/data/$USER/datasets} S1_MACHINE=pro6000
CFG=${1:?train config name}; EVTPL=${2:?eval template}
uv run --frozen --with huggingface_hub,flash-linear-attention python scripts/train-floor.py configs/$CFG.yaml
RUN=$(ls -td results/pro6000/$CFG/*/ | head -1); RUN=${RUN%/}
[ -d "$RUN/adapter" ] || { echo no_adapter; exit 1; }
sed -e "s#^name: .*#name: ev290_$CFG#" -e "s#adapter: .*#adapter: $RUN#" -e 's#name: frames$#name: foveated\n  frac: 0.333\n  upscale: 2\n  max_area: 0.045#' configs/$EVTPL.yaml > configs/ev290_$CFG.yaml
grep -q "adapter: $RUN" configs/ev290_$CFG.yaml || { echo inject_fail; exit 1; }
scripts/run.sh run configs/ev290_$CFG.yaml
echo CHAIN_DONE_$CFG
