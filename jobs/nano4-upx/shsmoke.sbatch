#!/bin/bash
#SBATCH --job-name=s1-shsmoke
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/u9117820/s1-logs/shsmoke-%j.out
set -euo pipefail
cd /work/u9117820/s1stream; source /work/u9117820/upx/lib.sh
module purge; module load cuda/13.0
$PY - <<'PYEOF'
import yaml
d = yaml.safe_load(open("configs/floor290_d3nano_c.yaml")); d["name"] = "sh_smoke"
d["backbone"] = {"name": "shieldstral", "asset": "shieldstral-3b", "memory_cap_gb": 40}
yaml.safe_dump(d, open("configs/sh_smoke.yaml", "w"), sort_keys=False)
PYEOF
$PY scripts/sh-smoke.py configs/sh_smoke.yaml 60 2>&1 | grep -v -E 'Loading weights|Warning|warn' | tail -30
echo SHSMOKE_DONE
