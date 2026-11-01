#!/usr/bin/env bash
# WiSE-FT eval chain (2026-11-01), run inside the gen container AFTER its arm-b 290 evals finish:
# derive alpha {0.25, 0.5, 0.75} interpolations of arm A (today's LoRA, the p. 52 run) and score each on both axes:
# UCF-290 (task axis) + jev-bench and MVBench (the two damaged generalization metrics). ~4 h per alpha for the 290.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
A_RUN=${A_RUN:-results/pro6000/ft_ucf_v1_d3nano_hard/20261023-200623}
DATA=/data/kidflash7011/datasets

until grep -q "FLOOR_EVAL_d3nano_DONE" floor-eval-d3nano-b.log 2>/dev/null; do sleep 300; done

for alpha in 0.5 0.25 0.75; do
    tag=${alpha/0./}
    out=results/pro6000/wise_a$tag
    uv run --frozen python scripts/wise-ft.py --run "$A_RUN" --asset d3-nano --alpha "$alpha" --out "$out"
    uv run --frozen python scripts/jev-bench.py --adapter "$out" --out "results/pro6000/jevbench_d3nano_wise$tag"
    uv run --frozen python scripts/video-mc.py --bench mvbench --root "$DATA/mvbench" \
        --adapter "$out" --out "results/pro6000/mvbench_d3nano_wise$tag"
    sed -e "s#^name: .*#name: wise290_d3nano_a$tag#" -e "s#adapter: .*#adapter: $out#" \
        configs/ft_ucf_v1_d3nano_eval.yaml > "configs/wise290_d3nano_a$tag.yaml"
    ls results/*/wise290_d3nano_a$tag/*/report.json >/dev/null 2>&1 || scripts/run.sh run "configs/wise290_d3nano_a$tag.yaml"
done
echo WISE_FT_DONE
