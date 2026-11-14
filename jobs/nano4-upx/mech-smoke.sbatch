#!/bin/bash
#SBATCH --job-name=s1-msmoke
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/work/u9117820/s1-logs/msmoke-%j.out
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
A=$(ls -d results/nano4/up290_floor_s0_frames/*/ | tail -1)
rm -rf results/nano4/mech-smoke; mkdir -p results/nano4/mech-smoke
$PY scripts/mech.py layers --config configs/viz_floor.yaml --events results/pro6000/jev-hard-events.jsonl --a $A --out results/nano4/mech-smoke/layers --smoke 2>&1 | grep --line-buffered -vE 'Warning|Loading weights|causal_conv1d'
head -c 1500 results/nano4/mech-smoke/layers/layers.jsonl; echo
# pooling modes on a 2-video config
$PY - <<'PY'
import yaml
d = yaml.safe_load(open("configs/viz_floor.yaml")); d["name"] = "mech_smoke"
d["dataset"]["n_anomaly"] = 1; d["dataset"]["n_normal"] = 1
yaml.safe_dump(d, open("configs/mech_smoke.yaml", "w"), sort_keys=False)
PY
for m in identity pool1x poolrep; do $PY scripts/mech.py pool --config configs/mech_smoke.yaml --mode $m --kernel bicubic --out results/nano4/mech-smoke/$m 2>&1 | grep --line-buffered -vE 'Warning|Loading weights|causal_conv1d' | tail -5; head -2 results/nano4/mech-smoke/$m/scores.jsonl; done
B=$(ls -d results/nano4/up290_floor_s0_all2x/*/ | head -1)
$PY - "$B" <<'PY'
import json, sys
ref = {(r["video"], r["window"]): r["score"] for r in map(json.loads, open(sys.argv[1] + "scores.jsonl"))}
rows = [json.loads(l) for l in open("results/nano4/mech-smoke/identity/scores.jsonl")]
d = [abs(r["score"] - ref[(r["video"], r["window"])]) for r in rows if (r["video"], r["window"]) in ref]
print("identity vs ordinary 2x run:", len(d), "windows, max |dp|", max(d), "mean", sum(d) / len(d))
PY
echo MSMOKE_DONE
