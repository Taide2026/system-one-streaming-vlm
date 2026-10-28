#!/usr/bin/env bash
# Generalization test chain (2026-10-28), run inside a container (scripts/docker.sh exec ... or nohup in it):
#   scripts/ucf101-chain.sh <rebuild log> <probe train log>
# Waits for the UCF101 rebuild (scripts/ucf101-from-hf.py, its log ends with "clips written to"), then runs the
# zero-shot and UCF-Crime-LoRA evals on UCF101; then waits for the linear-probe training (its log's "saved" line / the
# run folder's readout.safetensors), points configs/ucf101_d3nano_probe.yaml at that run and evaluates it too.
# Accuracy of each: scripts/ucf101-accuracy.py results/<machine>/ucf101_d3nano_*/<time>.
set -euo pipefail
cd "$(dirname "$0")/.."
REBUILD_LOG=$1; PROBE_LOG=$2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

until grep -q "clips written to" "$REBUILD_LOG"; do sleep 60; done
echo "rebuild done: $(tail -1 "$REBUILD_LOG")"
for cfg in ucf101_d3nano_zs ucf101_d3nano_ucfft; do
    scripts/run.sh run configs/$cfg.yaml
    uv run --frozen python scripts/ucf101-accuracy.py "$(ls -td results/*/$cfg/*/ | head -1)" --per-class 5
done

until [ -n "$(ls -d results/*/ft_ucf_v1_d3nano_probe/*/readout.safetensors 2>/dev/null)" ] && grep -q "calibration\|temperature" "$PROBE_LOG"; do sleep 60; done
run=$(dirname "$(ls -t results/*/ft_ucf_v1_d3nano_probe/*/readout.safetensors | head -1)")
echo "probe run: $run"
sed -i "s#adapter: results/pro6000/ft_ucf_v1_d3nano_probe/RUN#adapter: $run#" configs/ucf101_d3nano_probe.yaml
scripts/run.sh run configs/ucf101_d3nano_probe.yaml
uv run --frozen python scripts/ucf101-accuracy.py "$(ls -td results/*/ucf101_d3nano_probe/*/ | head -1)" --per-class 5
echo CHAIN_DONE
