#!/usr/bin/env bash
# Run-2 probe-fit for C, A, B in order on the GPU (lr 1e-3, 1000 steps). Features come from run-1.
R2="$(git rev-parse --show-toplevel)/results/probe_ucf_short_v1_f8/run-2"
cd "$(git rev-parse --show-toplevel)" || exit 1
for c in C A B; do
  avail=$(free -g | awk '/Mem:/{print $7}')
  echo "$(date -Iseconds) $c fit start, available ${avail} GB"
  if [ "$avail" -lt 30 ]; then echo "available memory below 30 GB, stop"; exit 2; fi
  /usr/bin/time -f "$c wall %e s, max rss %M KB" timeout 3h uv run python $R2/scripts/fit_gpu.py \
    $S1_FEATURES/probe_ucf_short_v1_f8/$c/run-1 $R2/fit-$c > $R2/logs/fit-$c.log 2>&1
  code=$?
  echo "$(date -Iseconds) $c fit exit $code"
  tail -n 2 $R2/logs/fit-$c.log
  if [ $code -ne 0 ]; then exit $code; fi
done
exit 0
