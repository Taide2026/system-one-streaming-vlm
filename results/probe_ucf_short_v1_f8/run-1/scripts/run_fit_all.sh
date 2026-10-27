#!/usr/bin/env bash
# Fit probes for C, A, B in order (CPU). Logs to fit-<c>.log in the scratchpad.
S="${TMPDIR:-/tmp}"
cd "$(git rev-parse --show-toplevel)" || exit 1
for c in C A B; do
  echo "$(date -Iseconds) $c fit start, available $(free -g | awk '/Mem:/{print $7}') GB"
  /usr/bin/time -f "$c wall %e s, max rss %M KB" timeout 4h uv run s1stream probe-fit $S1_FEATURES/probe_ucf_short_v1_f8/$c/run-1 > $S/fit-$c.log 2>&1
  code=$?
  echo "$(date -Iseconds) $c fit exit $code"
  tail -n 2 $S/fit-$c.log
  [ $code -ne 0 ] && exit $code
done
