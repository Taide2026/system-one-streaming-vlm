#!/bin/bash
#SBATCH --job-name=s1-oracle
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009
#SBATCH --time=10:00:00
#SBATCH --output=/work/u9117820/s1-logs/oracle-%j.out
# Oracle placement of the fine tokens (2026-11-12), 4 shards on 4 GPUs.
#   sbatch --export=ALL,MODEL=zs oracle.sbatch          zero-shot d3-nano
#   sbatch --export=ALL,MODEL=floor_s0 oracle.sbatch    fine-tuned seed 0
# A stratified sample of 3,000 windows (600 per group); every one of the 80 blocks is measured, see scripts/oracle.py.
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; hostname
L=$W/s1-logs; AD=""; [ "$MODEL" = floor_s0 ] && { fetch $F0; AD=$F0; }
name=oracle_$MODEL; mkeval $name "$AD" frames > /dev/null
A=$(for d in results/nano4/$( [ "$MODEL" = zs ] && echo up290_zs_1x || echo up290_floor_s0_frames )/*/; do [ -f $d/run.json ] && echo $d; done | tail -1)
OUT=results/nano4/oracle_$MODEL/$(date +%Y%m%d-%H%M%S)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $PY scripts/oracle.py --config configs/$name.yaml --events results/pro6000/jev-hard-events.jsonl --a $A --out $OUT --ks 5 24 --per-group 600 --shard $i --nshards 4 > $L/oracle-$MODEL-$i.log 2>&1 &
done
wait
cat $OUT/rows.shard*.jsonl > $OUT/rows.jsonl; wc -l $OUT/rows.jsonl; echo $OUT
echo ORACLE_JOB_DONE
