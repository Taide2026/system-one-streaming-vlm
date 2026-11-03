#!/usr/bin/env bash
# fov2h chain (2026-11-03), run INSIDE a container: scripts/fov2h-chain.sh <d3nano|qwen2b>.
# Mining (train normals, the model's own floor-C adapter) -> fov2h training -> UCF-290 eval.
# A file-based script because the docker-exec one-liners kept losing $() substitutions across quoting layers
# (four chains died at the same hand-off; adapter lines ended up empty and evals scored the base model).
set -euo pipefail
cd "$(dirname "$0")/.."
export S1_DATASETS=${S1_DATASETS:-/data/kidflash7011/datasets} S1_MACHINE=${S1_MACHINE:-pro6000}
M=${1:?usage: fov2h-chain.sh <d3nano|qwen2b>}
HIVAU=$S1_DATASETS/HolmesVAU/HIVAU-70k/raw_annotations/ucf_database_train.json
ROOT=/data/kidflash7011/s1-assets/ucf-crime

if [ "$M" = d3nano ]; then BB=d3; ASSET=d3-nano; EVTPL=ft_ucf_v1_d3nano_eval; else BB=qwen_first_token; ASSET=qwen3.5-2b; EVTPL=ft_ucf_v1_qwen2b_eval; fi
FLOOR=$(ls -td results/pro6000/floor_${M}_c/2026103*/ results/pro6000/floor_${M}_c/20261101*/ 2>/dev/null | head -1)
FLOOR=${FLOOR%/}
[ -d "$FLOOR/adapter" ] || { echo "no floor adapter under $FLOOR"; exit 1; }
MINED=results/pro6000/hardneg_train_${M}.jsonl

if [ ! -s "$MINED" ]; then
    uv run --frozen --with huggingface_hub python scripts/mine-hardneg.py --backbone "$BB" --asset "$ASSET" \
        --hivau "$HIVAU" --root "$ROOT" --adapter "$FLOOR" --out "$MINED"
fi

sed -e "s#^name: .*#name: fov2h_${M}_c#" \
    -e "s#extra_records: .*#extra_records: $MINED#" \
    "configs/fov2h_${M}_c.yaml" > /tmp/_c.yaml && cp /tmp/_c.yaml "configs/fov2h_${M}_c.yaml"
uv run --frozen --with huggingface_hub python scripts/train-floor.py "configs/fov2h_${M}_c.yaml"
RUN=$(ls -td "results/pro6000/fov2h_${M}_c"/*/ | head -1)
RUN=${RUN%/}
[ -d "$RUN/adapter" ] || { echo "training produced no adapter"; exit 1; }

sed -e "s#^name: .*#name: fov2h_290_${M}_c#" -e "s#adapter: .*#adapter: $RUN#" \
    -e "s#name: frames#name: foveated\n  frac: 0.333\n  upscale: 2\n  max_area: 0.045#" \
    "configs/$EVTPL.yaml" > "configs/fov2h_290_${M}_c.yaml"
grep -q "adapter: $RUN" "configs/fov2h_290_${M}_c.yaml" || { echo "adapter injection failed"; exit 1; }
scripts/run.sh run "configs/fov2h_290_${M}_c.yaml"
echo "FOV2H_${M}_CHAIN_DONE"
