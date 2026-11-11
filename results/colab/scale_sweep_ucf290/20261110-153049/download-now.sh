#!/usr/bin/env bash
# Download the finished Colab score files now, without waiting for the VM to stop (2026-11-11, for the interim artifact).
# Usage: download-now.sh <upscale> ...   e.g. bicubic1.75 bicubic3
set -uo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov/results/colab/scale_sweep_ucf290/20261110-153049
ROOT=1-hm9GQgErrT_ZfS7w4F2Byyb_0agu4EG   # Drive folder s1-colab-scale-score
child() {
  gws drive files list --params "{\"q\":\"'$1' in parents and name = '$2' and trashed = false\",\"fields\":\"files(id)\"}" 2>/dev/null |
    python3 -c 'import sys,json;s=sys.stdin.read();f=json.loads(s[s.find("{"):])["files"];print(f[0]["id"] if f else "")'
}
out=$(child $ROOT out)
for m in fov2 mixkl; do
  md=$(child $out $m)
  for u in "$@"; do
    d=$(child $md $u); mkdir -p out/$m/$u
    for f in scores.jsonl config.yaml; do
      id=$(child $d $f); [ -n "$id" ] && gws drive files get --params "{\"fileId\":\"$id\",\"alt\":\"media\"}" -o out/$m/$u/$f > /dev/null 2>&1
    done
    echo "$m $u $(wc -l < out/$m/$u/scores.jsonl) lines"
  done
done
