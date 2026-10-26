# Small control (2026-10-25): ctrl2b.sh ucf | xd, inside a kv-across container. ucf: train + eval on UCF 290, then the
# zero-shot UCF and XD references; xd: train + eval on XD 800.
cd /work
export PY=/opt/venv/bin/python S1_ASSETS=/data/kidflash7011/s1-assets S1_MACHINE=pro6000 S1_DATASETS=/data/kidflash7011/datasets
export HF_HOME=/data/kidflash7011/hf-cache OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OPENCV_FOR_THREADS_NUM=8
set -e
if [ "$1" = ucf ]; then
  scripts/stage2-run.sh qwen2b 4
  bash scripts/cross.sh zs_ucf_v1_qwen2b_eval ucf290-manifest.json 4
  bash scripts/cross.sh zs_xd_v1_qwen2b_eval xd800-manifest.json 4
else
  scripts/stage2-xd-run.sh qwen2b 4
fi
echo "CTRL2B $1 DONE"
