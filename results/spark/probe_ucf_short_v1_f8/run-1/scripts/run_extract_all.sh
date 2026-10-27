#!/bin/bash
# Real extraction of the three conditions, one after the other (decision 14).
cd "$(git rev-parse --show-toplevel)" || exit 1
LOG="${TMPDIR:-/tmp}"
for c in C A B; do
  avail=$(free -g | awk '/^Mem:/ {print $7}')
  echo "$(date -Is) $c start, available ${avail} GB"
  if [ "$avail" -lt 29 ]; then echo "$c: available memory below 29 GB, stop"; exit 2; fi
  /usr/bin/time -f "$c wall %e s, max rss %M KB" timeout 4h uv run s1stream probe-extract configs/probe_ucf_short_v1_f8.yaml --condition $c --run run-1 > $LOG/extract-$c.log 2>&1
  code=$?
  echo "$(date -Is) $c exit $code"; tail -2 $LOG/extract-$c.log
  if [ $code -ne 0 ]; then exit $code; fi
done
