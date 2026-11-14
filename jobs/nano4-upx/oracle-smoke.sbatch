#!/bin/bash
#SBATCH --job-name=s1-osmoke
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/u9117820/s1-logs/osmoke-%j.out
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
rm -rf results/nano4/oracle-smoke
A=$(ls -d results/nano4/up290_zs_f16/*/ | tail -1)
mkeval oracle_zs "" frames > /dev/null
SECONDS=0
$PY scripts/oracle.py --config configs/oracle_zs.yaml --events results/pro6000/jev-hard-events.jsonl --a $A --out results/nano4/oracle-smoke --ks 5 24 --per-group 600 --limit 12 2>&1 | grep --line-buffered -E 'shard|ORACLE|Error|Traceback' || true
echo "elapsed $SECONDS s for 12 windows (includes model load)"
$PY scripts/oracle-peek.py results/nano4/oracle-smoke/rows.shard0.jsonl
echo OSMOKE_DONE
