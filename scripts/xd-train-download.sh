#!/usr/bin/env bash
# Download the XD-Violence training videos (3,954 videos, five OneDrive zips from roc-ng.github.io/XD-Violence) on a
# server with curl only (2026-10-24). The anonymous OneDrive links need the cookie flow: open the share link once with a
# cookie jar (FedAuth), read the file path from the final URL, then fetch <host>/<path>?ga=1 with the jar.
#   scripts/xd-train-download.sh <download dir>        (resumable: curl -C -, parts run in parallel)
set -uo pipefail
out="${1:-/data/kidflash7011/xd-train-download}"; mkdir -p "$out"; cd "$out"
UA="Mozilla/5.0 (X11; Linux x86_64) Chrome/130"
host="https://stuxidianeducn-my.sharepoint.com"
links=(
  "$host/:u:/g/personal/pengwu_stu_xidian_edu_cn/ERiQQLUAex1CnwuBIEOK9YgBtcqU3hR_ZDtahCwLtVVkew?e=QHJ3bz"
  "$host/:u:/g/personal/pengwu_stu_xidian_edu_cn/EU-CqR-yYmVEqlFR1qHs3ZwB_Xfmi9d2augzSoOgDM66ug?e=gNNLIR"
  "$host/:u:/g/personal/pengwu_stu_xidian_edu_cn/EVcHk8zeeQJGhoDrsj1I-RsBXwGgOwrr_qU_8Yvx1KVnEw?e=84ftkC"
  "$host/:u:/g/personal/pengwu_stu_xidian_edu_cn/EV9x2SOsTWNFjAAukYmTVIEBKTFz2rc_lkJ4N-8Y7AbIvA?e=fOZU2d"
  "$host/:u:/g/personal/pengwu_stu_xidian_edu_cn/ERGkxZqFDJlGs4GM1CvhOY4B5ASIKQIS4elt3VWZqjR65g?e=jdbmcO"
)
fetch() {  # fetch <index> <share link>
  local i="$1" link="$2" jar="jar$1" final path name
  final=$(curl -sS -m 120 -L -c "$jar" -b "$jar" -A "$UA" -o /dev/null -w "%{url_effective}" "$link") || { echo "part$i: share page failed"; return 1; }
  path=$(python3 -c "import sys, urllib.parse as u; q=u.parse_qs(u.urlparse(sys.argv[1]).query); print(q['id'][0])" "$final") || { echo "part$i: no id in $final"; return 1; }
  name=$(basename "$path")
  echo "part$i: $name ($path)"
  curl -sS -L -b "$jar" -A "$UA" --retry 20 --retry-delay 30 --retry-all-errors -C - -o "$name" "$host$path?ga=1" \
    && echo "part$i: done $name $(stat -c %s "$name") bytes" || echo "part$i: FAILED $name"
}
pids=()
for i in "${!links[@]}"; do fetch "$i" "${links[$i]}" & pids+=($!); sleep 5; done
for pid in "${pids[@]}"; do wait "$pid"; done
ls -la "$out"/*.zip
echo XD_DOWNLOAD_DONE
