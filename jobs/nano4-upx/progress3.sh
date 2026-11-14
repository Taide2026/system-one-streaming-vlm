#!/bin/bash
date '+%H:%M'; squeue -u u9117820 -h -o '%i %j %T %M %l %r'
cd /work/u9117820/s1stream/results/nano4
prog() { local tot=$1; shift; for n in "$@"; do d=$(ls -td $n/*/ 2>/dev/null | head -1); if [ -z "$d" ]; then echo "$n 0"; elif [ -f ${d}run.json ]; then echo "$n 100"; else echo "$n $(( 100 * $(wc -l < ${d}scores.jsonl 2>/dev/null || echo 0) / tot ))"; fi; done; }
prog 16515 cvad83_zs_v180_pboxes cvad83_f0_v180_pboxes
prog 18390 up290_floor_s0_bl_pboxes up290_floor_s1_bl_pboxes up290_floor_s2_bl_pboxes up290_zs_bl_pboxes up290_floor_s0_bl_prect_reuse up290_floor_s1_bl_prect_reuse up290_floor_s2_bl_prect_reuse q9b_zs_1x q9b_zs_2x q2b_zs_05x q2b_zs_1x q2b_zs_2x
L=/work/u9117820/s1-logs
grep -h -E '^shieldstral|SHLAT' $L/six-771911-g0.log 2>/dev/null | cut -c1-170
grep -h -E 'FAILED|Traceback|Error' $L/boxes-771611*.{out,log} $L/six-771911*.{out,log} $L/six-772054*.{out,log} 2>/dev/null | sort | uniq -c | head -4
