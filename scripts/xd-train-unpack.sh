#!/usr/bin/env bash
# Unpack the five XD-Violence training zips into a flat $S1_ASSETS/xd-violence/train-videos/<name>.mp4 (2026-10-24) and
# check the names against HIVAU-70k's xd_database_train.json. Test videos stay in videos/ (the dataset loader globs
# only that folder). scripts/xd-train-unpack.sh <download dir> <asset root> <hivau xd json>
set -euo pipefail
dl="$1"; root="$2"; json="$3"; dest="$root/train-videos"; mkdir -p "$dest" "$dl/unpacked"
for z in "$dl"/*.zip; do echo "unzip $z"; unzip -q -o "$z" -d "$dl/unpacked"; done
find "$dl/unpacked" -type f -name "*.mp4" -exec mv -n {} "$dest"/ \;
echo "train videos: $(ls "$dest" | wc -l)"
python3 - "$dest" "$json" <<'PYIN'
import json, os, sys
dest, names = sys.argv[1], json.load(open(sys.argv[2]))
have = {f[:-4] for f in os.listdir(dest) if f.endswith(".mp4")}
missing = [n for n in names if n not in have]
print(f"{len(names)} annotated, {len(have)} files, {len(missing)} missing")
print("\n".join(missing[:10]))
PYIN
echo XD_UNPACK_DONE
