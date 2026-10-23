#!/usr/bin/env bash
# Sharded evaluation on a shared machine (2026-10-24): the 290-video eval config split into N processes on the same
# GPU, each over a disjoint video list (dataset.names), then merged (scripts/merge-shards.py) and reported.
#
#   scripts/eval-shards.sh <eval config.yaml> <manifest.json with the 290 videos> <out dir> [N=3]
#
# Needs PY, S1_ASSETS, S1_MACHINE from the caller (scripts/ft.sh sets them); S1_PREFETCH applies to each shard.
# The runner measures per-window time, which is not a streaming latency here (N processes share the GPU and CPU).
set -euo pipefail
config="$1"; manifest="$2"; out="$3"; n="${4:-3}"
PY="${PY:-/opt/venv/bin/python}"
export PYTHONPATH=src S1_PREFETCH="${S1_PREFETCH:-4}"
# A shared 128-core box: without a cap every torch process spawns one thread per core and they fight.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p "$out"
$PY - "$config" "$manifest" "$out" "$n" <<'EOF'
import json, sys, yaml
from pathlib import Path
config, manifest, out, n = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
names = [video["name"] for video in json.loads(manifest.read_text())]
data = yaml.safe_load(config.read_text())
for k in range(n):
    shard = dict(data)
    shard["dataset"] = {key: value for key, value in data["dataset"].items()
                        if key not in ("n_anomaly", "n_normal", "max_duration_s", "seed", "min_duration_s")}
    shard["dataset"]["names"] = names[k::n]
    (out / f"shard{k}.yaml").write_text(yaml.safe_dump(shard, sort_keys=False))
print(f"{len(names)} videos into {n} shards")
EOF
pids=()
for k in $(seq 0 $((n - 1))); do
    $PY -m s1stream.cli run "$out/shard$k.yaml" --out "$out/shard$k" > "$out/shard$k.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
$PY scripts/merge-shards.py "$out/merged" "$out"/shard[0-9]*/
$PY -m s1stream.cli report "$out/merged"
$PY - "$out/merged/report.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{d['windows']} windows: frame AUC {d['auc']['frame']:.4f}, frame AP {d['ap']['frame']:.4f}")
EOF
