#!/usr/bin/env bash
# The latency re-measurement pass on pro6000 (2026-10-24): every V1 UCF-Crime model of main, sampled (the first 2
# windows of each of the 290 test videos, 5 warm-up windows left out), in a container whose venv has
# flash-linear-attention, on one GPU. Compared with the full run's report.json when one exists. Run after the
# evaluation queue has drained and when nothing else computes on the GPU (nvidia-smi).
#
#   MAIN=/data/<user>/system-one-streaming-vlm scripts/latency-pass.sh [config stem ...]
#     default stems: d3_flash d3_nano clef_flash startlux_4b startlux_9b h2o_lightning_4b d1_3b qwen3_5_9b
#     output: $MAIN/results/latency-fla/<stem>/   (main's results tree; the branch keeps the scripts only)
#
# Decision rule: a model whose sampled P95 differs from its full run by more than 10 % and 100 ms gets a full
# re-run on the fast kernels (scripts/run.sh run <config> inside the same container).
set -euo pipefail
MAIN="${MAIN:?set MAIN=<main checkout>}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/venv/bin/python}"
export S1_ASSETS="${S1_ASSETS:-/data/$(id -un)/s1-assets}" S1_MACHINE="${S1_MACHINE:-pro6000}"
stems=("$@")
[ ${#stems[@]} -gt 0 ] || stems=(d3_flash d3_nano clef_flash startlux_4b startlux_9b h2o_lightning_4b d1_3b qwen3_5_9b)
$PY -c "import fla" || { echo "flash-linear-attention is not installed in $PY" >&2; exit 1; }
cd "$MAIN"
for stem in "${stems[@]}"; do
    config="configs/${stem}_ucf_test_v1_f8.yaml"
    [ -f "$config" ] || { echo "no $config, skipping"; continue; }
    full=$(ls -d "results/pro6000/${stem}_ucf_test_v1_f8"/*/ 2>/dev/null | tail -1 || true)
    out="results/latency-fla/$stem/$(date +%Y%m%d-%H%M%S)"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    PYTHONPATH=src $PY "$HERE/scripts/latency-sample.py" "$config" --out "$out" ${full:+--compare "${full%/}"}
done
