#!/bin/bash
#SBATCH --job-name=s1-sixb
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:3
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=08:00:00
#SBATCH --output=/work/u9117820/s1-logs/six-%j.out
# Three further analyses (2026-11-13 night), UCF-Crime 290:
#   (3) model size: Shieldstral (3.85B) latency by stage, with tokens cut after the tower and with key frames + crops.
#   (4) decision model against the plain VLM it was built from: Qwen3.5-2B base (d3-nano's base) at 0.5x, 1x, 2x and
#       Qwen3.5-9B base (d3-flash's base) at 1x, 2x, read zero-shot from the first-token logits.
#   (6) streaming reuse: the rectangle in place, with frame 0 of each window taken from the previous window's frame 7
#       instead of being encoded (three fine-tuned seeds).
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
L=$W/s1-logs; J=$SLURM_JOB_ID; fetch $F0 $F1 $F2
done_already() { for d in results/nano4/$1/*/; do [ -f $d/run.json ] && return 0; done 2>/dev/null; return 1; }
base() {  # $1 name, $2 asset, $3 form
  done_already $1 && return 0
  $PY - "$1" "$2" "$(form_yaml $3)" <<'PYEOF'
import sys, yaml
name, asset, block = sys.argv[1:]
d = yaml.safe_load(open("configs/floor290_d3nano_c.yaml")); d["name"] = name
d["backbone"] = {"name": "qwen_first_token", "asset": asset, "memory_cap_gb": 60}
d["input_form"] = yaml.safe_load(block)
yaml.safe_dump(d, open(f"configs/{name}.yaml", "w"), sort_keys=False)
PYEOF
  $PY -m s1stream.cli run configs/$1.yaml --out results/nano4/$1/$(date +%Y%m%d-%H%M%S) && echo "DONE $1"
}
reuse() {  # $1 seed tag, $2 adapter
  local name=up290_floor_$1_bl_prect_reuse; done_already $name && return 0
  mkeval $name $2 frames > /dev/null
  $PY scripts/bl.py eval --config configs/$name.yaml --out results/nano4/$name/$(date +%Y%m%d-%H%M%S) --mode patch-rect --key 0 7 --reuse-key 1 2>&1 | grep --line-buffered -E 'windows|BL_DONE|Error|Traceback' && echo "DONE $name"
}
export -f done_already base reuse
( export CUDA_VISIBLE_DEVICES=0; base q2b_zs_1x qwen3.5-2b frames ) > $L/six-$J-g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; base q2b_zs_2x qwen3.5-2b up:2 ) > $L/six-$J-g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; base q2b_zs_05x qwen3.5-2b up:0.5:area ) > $L/six-$J-g2.log 2>&1 &
wait; grep -h -E '^DONE|FAILED|Traceback|Error|^shieldstral|SHLAT' $L/six-$J-g*.log | cut -c1-200 || true
$PY scripts/evalrun.py --events results/pro6000/jev-hard-events.jsonl q2b_zs_05x q2b_zs_1x q2b_zs_2x q9b_zs_1x q9b_zs_2x up290_floor_s0_bl_prect_reuse up290_floor_s1_bl_prect_reuse up290_floor_s2_bl_prect_reuse | cut -c1-128
echo SIX_DONE
