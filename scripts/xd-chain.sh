#!/usr/bin/env bash
# pro6000 host-side chain (2026-10-24): wait for the XD-Violence training download, unpack it into the asset, then start
# the XD fine-tunes in the two idle containers: d3-nano in s1stream-kidflash7011-ft2 (GPU 7, 4 eval shards) and the
# Qwen3.5-9B control in s1stream-kidflash7011-ev (GPU 8, 2 eval shards). Logs: chainF.log (d3-nano), chainG.log (Qwen).
#   nohup scripts/xd-chain.sh > xd-chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
dl=/data/kidflash7011/xd-train-download
until grep -q XD_DOWNLOAD_DONE "$dl/download.log"; do sleep 120; done
if grep -q FAILED "$dl/download.log"; then echo "download FAILED, not unpacking"; grep FAILED "$dl/download.log"; exit 1; fi
scripts/xd-train-unpack.sh "$dl" /data/kidflash7011/s1-assets/xd-violence /data/kidflash7011/datasets/HolmesVAU/HIVAU-70k/raw_annotations/xd_database_train.json || exit 1
docker exec -d s1stream-kidflash7011-ft2 bash -c "cd /work && scripts/stage2-xd-run.sh d3nano 4 > chainF.log 2>&1; echo CHAIN F DONE >> chainF.log"
docker exec -d s1stream-kidflash7011-ev bash -c "cd /work && scripts/stage2-xd-run.sh qwen 2 > chainG.log 2>&1; echo CHAIN G DONE >> chainG.log"
echo "XD chains launched $(date -u +%FT%TZ)"
