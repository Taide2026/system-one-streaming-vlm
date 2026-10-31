#!/usr/bin/env bash
# Training-floor gate evals (2026-10-31): scripts/floor-eval.sh <d3nano|qwen2b>, run inside that model's container.
# For each floor arm (b = multi-task mix, c = +KL anchor, d = DPO) the chain evaluates the newest
# results/pro6000/floor_<model>_<arm>/<time> adapter:
#   d3nano: quick generalization suite first for every arm (jev-bench acc+ECE, MVBench, Video-MME short,
#           UCF101 101-way accuracy, V1-on-UCF101 false alarms, UBnormal), then the two long UCF-290 streaming
#           passes per arm (V1 Yes/No AUC/AP and c14 14-way classification).
#   qwen2b: the UCF-290 V1 + c14 passes per arm (the cross-model trend gates; the MC suite plumbing is d3-only).
# Config-based evals are skipped when a finished run folder already exists, so the script is re-runnable.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:?usage: floor-eval.sh <d3nano|qwen2b>}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
DATA=/data/kidflash7011/datasets

run_dir() { ls -td "results/pro6000/floor_${MODEL}_$1"/*/ | head -1 | sed 's#/$##'; }

mkcfg() { # <template> <new name> <adapter run dir> [prompt]
    sed -e "s#^name: .*#name: $2#" -e "s#adapter: .*#adapter: $3#" "configs/$1.yaml" > "configs/$2.yaml"
    [ -n "${4:-}" ] && sed -i "s#prompt: .*#prompt: $4#" "configs/$2.yaml" || true
}

cfg_run() { # run configs/$1.yaml unless a finished run exists (records + report written => report.json present)
    if ls results/*/"$1"/*/report.json >/dev/null 2>&1; then echo "$1 done already, skipped"; return; fi
    scripts/run.sh run "configs/$1.yaml"
}

if [ "$MODEL" = d3nano ]; then
    for arm in b c d; do
        RUN=$(run_dir "$arm"); echo "== quick suite: d3nano $arm -> $RUN"
        uv run --frozen python scripts/jev-bench.py --adapter "$RUN" --out "results/pro6000/jevbench_d3nano_floor$arm"
        uv run --frozen python scripts/video-mc.py --bench mvbench --root "$DATA/mvbench" \
            --adapter "$RUN" --out "results/pro6000/mvbench_d3nano_floor$arm"
        uv run --frozen python scripts/video-mc.py --bench videomme --root "$DATA/videomme" --subset short \
            --adapter "$RUN" --out "results/pro6000/videomme_short_d3nano_floor$arm"
        mkcfg ucf101_d3nano_ucfft "ucf101_d3nano_floor$arm" "$RUN"
        cfg_run "ucf101_d3nano_floor$arm"
        uv run --frozen python scripts/ucf101-accuracy.py "$(ls -td results/*/ucf101_d3nano_floor$arm/*/ | head -1)" || true
        mkcfg ucf101_v1_d3nano_ucfft "ucf101_v1_d3nano_floor$arm" "$RUN"
        cfg_run "ucf101_v1_d3nano_floor$arm"
        mkcfg ubnormal_v1_d3nano_ucfft "ubnormal_v1_d3nano_floor$arm" "$RUN"
        cfg_run "ubnormal_v1_d3nano_floor$arm"
    done
    TPL=ft_ucf_v1_d3nano_eval
else
    TPL=ft_ucf_v1_qwen2b_eval
fi

for arm in b c d; do
    RUN=$(run_dir "$arm"); echo "== UCF-290: $MODEL $arm -> $RUN"
    mkcfg "$TPL" "floor290_${MODEL}_$arm" "$RUN"
    cfg_run "floor290_${MODEL}_$arm"
    mkcfg "$TPL" "floorc14_${MODEL}_$arm" "$RUN" c14
    cfg_run "floorc14_${MODEL}_$arm"
done
echo "FLOOR_EVAL_${MODEL}_DONE"
