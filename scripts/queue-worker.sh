#!/bin/bash
# A run queue for the Docker containers of scripts/docker.sh, one worker per container (developed on pro6000):
#   DATA=/data/$USER scripts/queue-worker.sh <container>     (start it with setsid nohup; one per container)
# Workers share $DATA/queue.txt (config names, one per line, # comments); a
# worker claims the first config that no one has claimed and whose assets are in place (fetch --check; ucf-crime:
# $DATA/ucf-setup.log says UCF SETUP DONE), under a lock, and runs it in its container into results/$S1_MACHINE/. Downloads still running only
# delay a config. Claimed configs go to queue.claimed, finished or twice-failed ones to queue.done; edit queue.txt any time.
set -uo pipefail
C=${1:?usage: queue-worker.sh <container>}
D=${DATA:-/data/$(id -un)}
MACHINE=${S1_MACHINE:-pro6000}
Q=$D/queue.txt DONE=$D/queue.done CLAIMED=$D/queue.claimed READY=$D/queue.ready LOG=$D/queue.log LOCK=$D/queue.lock
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p $D/logs; touch $DONE $CLAIMED $READY
log() { echo "$(date '+%F %T') [$C] $*" >> $LOG; }
ready() {
    local a
    for a in $(sed -n 's/^ *asset: *//p' "$REPO/configs/$1.yaml"); do
        grep -q -x -F "$a" $READY && continue
        if [ "$a" = ucf-crime ]; then
            grep -q "UCF SETUP DONE" $D/ucf-setup.log 2>/dev/null || return 1
        else
            docker exec $C uv run --frozen s1stream fetch --check "$a" >/dev/null 2>&1 || return 1
        fi
        echo "$a" >> $READY
    done
}
claim() {      # print the claimed config, or nothing
    exec 9>$LOCK; flock 9
    local c
    for c in $(grep -v -E '^\s*(#|$)' $Q | grep -v -x -F -f $CLAIMED); do
        if ready "$c"; then echo "$c" >> $CLAIMED; echo "$c"; break; fi
    done
    flock -u 9
}
log "worker started"
while :; do
    if [ -z "$(grep -v -E '^\s*(#|$)' $Q | grep -v -x -F -f $CLAIMED)" ]; then log "nothing left to claim"; break; fi
    next=$(claim)
    [ -z "$next" ] && { sleep 120; continue; }
    log "start $next"
    for attempt in 1 2; do
        if docker exec -e S1_MACHINE=$MACHINE $C scripts/run.sh run "configs/$next.yaml" > "$D/logs/$next-$attempt.log" 2>&1; then
            log "done $next (attempt $attempt)"; break
        fi
        log "FAIL $next (attempt $attempt): $(grep -a -E 'Error|error' "$D/logs/$next-$attempt.log" | tail -1)"
    done
    echo "$next" >> $DONE
done
