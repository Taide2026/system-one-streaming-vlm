#!/bin/bash
date '+%H:%M'; squeue -u u9117820 -h -o '%i %j %T %M %l'
cd /work/u9117820/s1stream/results/nano4
prog() { local tot=$1; shift; for n in "$@"; do d=$(ls -td $n/*/ 2>/dev/null | head -1); if [ -z "$d" ]; then echo "$n 0"; elif [ -f ${d}run.json ]; then echo "$n 100"; else echo "$n $(( 100 * $(wc -l < ${d}scores.jsonl 2>/dev/null || echo 0) / tot ))"; fi; done; }
prog 18390 up290_floor_s0_bl_prect up290_floor_s1_bl_prect up290_floor_s2_bl_prect up290_zs_bl_prect up290_floor_s0_bl_prect3 up290_floor_s1_bl_prect3 up290_floor_s2_bl_prect3 fl290_zs_prect fl290_s1_prect fl290_s2_prect
L=/work/u9117820/s1-logs
grep -h -E 'FAILED|Traceback|Error|CANCELLED|TIME LIMIT' $L/final2-770040*.{out,log} $L/final3-770047*.{out,log} 2>/dev/null | sort | uniq -c | head -5
grep -h -E '^(reference|fla) ' $L/final2-770040.out $L/final3-770047-g3.log 2>/dev/null | cut -c1-200
