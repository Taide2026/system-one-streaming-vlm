#!/usr/bin/env bash
# Stage 1 of the fine-tuning study on spark (2026-10-24). Uses the main checkout's venv (no uv sync).
#
#   scripts/ft.sh smoke            10 optimizer steps of configs/ft_ucf_v1_clef.yaml into results/smoke/ft-<time>
#   scripts/ft.sh train            the full run into results/<machine>/ft_ucf_v1_clef/<time> (start it in tmux)
#   scripts/ft.sh eval [run dir]   the fine-tuned model on all 290 test videos (configs/ft_ucf_v1_clef_eval.yaml with
#                                  backbone.adapter = run dir; default: the latest training run), then the report
#   scripts/ft.sh zeroshot         the zero-shot reference on the same machine (configs/zs_ucf_v1_clef_eval.yaml)
#   scripts/ft.sh stage1           smoke, train, eval, zeroshot in turn (one tmux session: `tmux new -s ft scripts/ft.sh stage1`)
set -euo pipefail
cd "$(dirname "$0")/.."
export S1_ASSETS="${S1_ASSETS:-$HOME/s1-assets}"
export S1_MACHINE="${S1_MACHINE:-spark}"
export PYTHONPATH=src
PY="${PY:-$HOME/system-one-streaming-vlm/.venv/bin/python}"
TRAIN=configs/ft_ucf_v1_clef.yaml

latest_run() { ls -d "results/$S1_MACHINE/ft_ucf_v1_clef"/*/ 2>/dev/null | tail -1; }

smoke() {
    local out="results/smoke/ft-$(date +%Y%m%d-%H%M%S)"
    $PY -m s1stream.cli train $TRAIN --out "$out" --limit-steps 10
    echo "smoke run: $out"; tail -3 "$out/train.log"
}

train() {
    $PY -m s1stream.cli train $TRAIN --out "results/$S1_MACHINE/ft_ucf_v1_clef/$(date +%Y%m%d-%H%M%S)"
}

eval_run() {
    local run="${1:-$(latest_run)}" out config
    [ -n "$run" ] || { echo "no training run found" >&2; exit 1; }
    run="${run%/}"
    config="$run/eval-config.yaml"
    sed "s#ADAPTER_RUN_DIR#$run#" configs/ft_ucf_v1_clef_eval.yaml > "$config"
    out="results/$S1_MACHINE/ft_ucf_v1_clef_eval/$(date +%Y%m%d-%H%M%S)"
    $PY -m s1stream.cli run "$config" --out "$out"
    $PY -m s1stream.cli report "$out"
    $PY - "$out/report.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); lat = d["latency"]
print(f"{d['windows']} windows: frame AUC {d['auc']['frame']:.4f}, frame AP {d['ap']['frame']:.4f}, "
      f"P95 {lat['p95_ms']:.0f} ms, RTF {lat['rtf_p95']:.2f}")
EOF
}

zeroshot() {
    local out="results/$S1_MACHINE/zs_ucf_v1_clef_eval/$(date +%Y%m%d-%H%M%S)"
    $PY -m s1stream.cli run configs/zs_ucf_v1_clef_eval.yaml --out "$out"
    $PY -m s1stream.cli report "$out"
}

case "${1:-}" in
    smoke) smoke ;;
    train) train ;;
    eval) eval_run "${2:-}" ;;
    zeroshot) zeroshot ;;
    stage1) smoke; train; eval_run; zeroshot ;;
    *) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
