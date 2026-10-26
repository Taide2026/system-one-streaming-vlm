# Cross-dataset evals (2026-10-25): cross.sh <config name> <manifest> <shards>; run inside a kv-across container.
cd /work; export PY=/opt/venv/bin/python S1_ASSETS=/data/kidflash7011/s1-assets S1_MACHINE=pro6000 PYTHONPATH=src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OPENCV_FOR_THREADS_NUM=8
out=results/pro6000/$1/$(date +%Y%m%d-%H%M%S)-shards
scripts/eval-shards.sh configs/$1.yaml $2 $out $3 && echo "CROSS $1 DONE"
