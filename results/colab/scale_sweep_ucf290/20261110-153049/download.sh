#!/usr/bin/env bash
# Download the 6 Colab score files and logs from Drive s1-colab-scale-score/out/ to ./out/ (2026-11-11).
# Waits until watch.log says the VM stopped (all 6 files complete on Drive). Cap 20 h.
set -uo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov/results/colab/scale_sweep_ucf290/20261110-153049
ROOT=1-hm9GQgErrT_ZfS7w4F2Byyb_0agu4EG   # Drive folder s1-colab-scale-score
for i in $(seq 1 240); do grep -q "VM stopped" watch.log && break; sleep 300; done
grep -q "VM stopped" watch.log || { echo "VM not stopped after 20 h, stop"; exit 1; }

child() {   # id of the child named $2 in Drive folder $1
  gws drive files list --params "{\"q\":\"'$1' in parents and name = '$2' and trashed = false\",\"fields\":\"files(id)\"}" 2>/dev/null |
    python3 -c 'import sys,json;s=sys.stdin.read();f=json.loads(s[s.find("{"):])["files"];print(f[0]["id"] if f else "")'
}
out=$(child $ROOT out)
for m in fov2 mixkl; do
  for u in bicubic1.5 bicubic1.75 bicubic3; do
    d=$(child "$(child $out $m)" $u); mkdir -p out/$m/$u
    for f in scores.jsonl config.yaml; do
      id=$(child $d $f); [ -n "$id" ] || { echo "missing $m/$u/$f"; continue; }
      gws drive files get --params "{\"fileId\":\"$id\",\"alt\":\"media\"}" -o out/$m/$u/$f > /dev/null 2>&1
    done
    echo "$m $u $(wc -l < out/$m/$u/scores.jsonl) lines"
  done
done
echo "download done $(date -u +%FT%TZ)"
