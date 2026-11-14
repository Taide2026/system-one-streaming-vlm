#!/bin/bash
# Percent complete of today's evaluations on nano4 (2026-11-13). Run on nano4: bash /work/u9117820/upx/progress.sh
date '+%H:%M'; squeue -u u9117820 -h -o '%i %j %T %M %l'
cd /work/u9117820/s1stream/results/nano4
prog() { local tot=$1; shift; for n in "$@"; do d=$(ls -td $n/*/ 2>/dev/null | head -1); if [ -z "$d" ]; then echo "$n 0"; elif [ -f ${d}run.json ]; then echo "$n 100"; else echo "$n $(( 100 * $(wc -l < ${d}scores.jsonl 2>/dev/null || echo 0) / tot ))"; fi; done; }
prog 26810 cvadall_zs_v180 cvadall_f0_v180 cvadall_zs_v180up2 cvadall_f0_v180up2 cvadall_zs_v360 cvadall_f0_v360
prog 10295 cvad30_zs_v180_pdiff cvad30_f0_v180_pdiff cvad30_zs_v180_ptgm cvad30_f0_v180_ptgm cvad30_zs_v180_dtd cvad30_f0_v180_dtd cvad30_zs_v180_pool cvad30_f0_v180_pool
prog 18390 fl290_zs_kc3 fl290_s1_1p5 up290_zs_dn075 up290_floor_s0_dn075 up290_floor_s1_dn075 up290_floor_s2_dn075 up290_floor_s0_bl_w1x up290_floor_s1_bl_w1x up290_floor_s2_bl_w1x
L=/work/u9117820/s1-logs
grep -h -E 'FAILED|Traceback|Error|CANCELLED|TIME LIMIT' $L/cvad-769312*.{out,log} $L/blind-769377*.{out,log} $L/flash-768728*.{out,log} $L/ucfdown-769296*.{out,log} 2>/dev/null | sort | uniq -c | head -6
[ -f $L/blind-769377-cover.log ] && tail -14 $L/blind-769377-cover.log | cut -c1-230
