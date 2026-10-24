#!/usr/bin/env bash
# Unpack the five XD-Violence training zips into a flat $S1_ASSETS/xd-violence/train-videos/<name>.mp4 (2026-10-24) and
# check the names against HIVAU-70k's xd_database_train.json. Test videos stay in videos/ (the dataset loader globs
# only that folder). The zips hold a few corrupt entries (bad CRC / invalid compressed data): unzip's warnings are
# collected, those files are deleted, and training skips them (data.skip_missing).
#   scripts/xd-train-unpack.sh <download dir> <asset root> <hivau xd json>
set -uo pipefail
dl="$1"; root="$2"; json="$3"; dest="$root/train-videos"; mkdir -p "$dest" "$dl/unpacked"
: > "$dl/unzip-errors.log"
for z in "$dl"/*.zip; do echo "unzip $z"; unzip -q -o "$z" -d "$dl/unpacked" 2>&1 | tee -a "$dl/unzip-errors.log" | grep -c . | sed "s/^/  warning lines: /"; done
grep -oE "$dl/unpacked/[^ ]+\.mp4" "$dl/unzip-errors.log" | sort -u > "$dl/corrupt.txt"
echo "corrupt entries: $(wc -l < "$dl/corrupt.txt")"
while read -r f; do rm -f "$f"; done < "$dl/corrupt.txt"
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
