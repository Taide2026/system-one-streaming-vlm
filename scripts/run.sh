#!/usr/bin/env bash
# Run an s1stream experiment on a GPU machine (developed on spark-20cb, NVIDIA GB10).
#
#   scripts/run.sh setup              install uv when missing, uv sync, CPU tests
#   scripts/run.sh assets             jev-omni with `s1stream fetch`; ucf-crime built from the ucf-clips download
#   scripts/run.sh smoke  [config]    5 windows into results/smoke/<time>, then report
#   scripts/run.sh run    [config]    full run into results/<config name>/<time>, then report
#   scripts/run.sh report <run dir>   metrics only, no GPU
#   scripts/run.sh all    [config]    setup, assets, smoke, run
#
# Environment:
#   S1_ASSETS  asset root (default ~/s1-assets)
#   UCF_CLIPS  extracted bear7011/ucf-clips: labels.csv, short/, medium/, long/ (default ~/datasets/ucf-clips)
# A full run takes a while: start it inside tmux.
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export S1_ASSETS="${S1_ASSETS:-$HOME/s1-assets}"
UCF_CLIPS="${UCF_CLIPS:-$HOME/datasets/ucf-clips}"
DEFAULT_CONFIG=configs/jev_omni_ucf20.yaml

# ucf-clips has the 1900 UCF-Crime videos but no event times. The annotation file comes from the
# UCF-Crime authors' repository; its SHA-256 equals the one locked in assets.yaml.
ANNOTATION=annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
ANNOTATION_URL=https://raw.githubusercontent.com/WaqasSultani/AnomalyDetectionCVPR2018/master/Temporal_Anomaly_Annotation.txt

setup() {
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    # Triton compiles a C helper at run time and needs Python.h. A uv-managed Python ships it;
    # the system Python does not unless python3.12-dev is installed (needs root).
    uv sync --managed-python
    uv run python -c "import os, sysconfig; assert os.path.isfile(sysconfig.get_paths()['include'] + '/Python.h'), 'no Python.h'"
    uv run pytest -q
}

# Build $S1_ASSETS/ucf-crime in the layout of the README: symlinks into $UCF_CLIPS, no copies.
ucf_crime() {
    local dest="$S1_ASSETS/ucf-crime" want have
    [ -f "$UCF_CLIPS/labels.csv" ] || { echo "No labels.csv in $UCF_CLIPS (set UCF_CLIPS)" >&2; exit 1; }
    mkdir -p "$dest/annotations"
    curl -sfL "$ANNOTATION_URL" -o "$dest/$ANNOTATION"
    want=$(uv run python -c "from s1stream.assets import LOCK_FILE, load_lock
print(load_lock(LOCK_FILE)['ucf-crime'].sha256['$ANNOTATION'])")
    have=$(sha256sum "$dest/$ANNOTATION" | cut -d' ' -f1)
    [ "$have" = "$want" ] || { echo "Annotation SHA-256 mismatch: $have" >&2; exit 1; }
    uv run python - "$UCF_CLIPS" "$dest" <<'EOF'
import csv
import sys
from pathlib import Path

clips, dest = Path(sys.argv[1]), Path(sys.argv[2])
linked, missing = 0, []
with (clips / "labels.csv").open() as labels:
    for row in csv.DictReader(labels):
        source = clips / row["file"]
        if not source.is_file():
            missing.append(row["file"])
            continue
        folder = row["class"]
        if folder == "Normal":
            folder = "Testing_Normal_Videos_Anomaly" if row["split"] == "test" else "Training_Normal_Videos_Anomaly"
        link = dest / "videos" / folder / source.name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.unlink(missing_ok=True)
        link.symlink_to(source)
        linked += 1
print(f"ucf-crime: {linked} videos linked into {dest}/videos")
if missing:
    sys.exit(f"{len(missing)} videos of labels.csv are missing, e.g. {missing[0]} (is the extraction done?)")
EOF
}

assets() {
    mkdir -p "$S1_ASSETS"
    uv run s1stream fetch jev-omni
    ucf_crime
}

report() {
    uv run s1stream report "$1"
    uv run python - "$1/report.json" <<'EOF'
import json
import sys

data = json.load(open(sys.argv[1]))
lat = data["latency"]
print(f"{data['windows']} windows, {data['videos']} videos, peak GPU {data['peak_gpu_gb']:.1f} GB, load {data['load_s']:.0f} s")
print(f"latency P50/P95/P99 {lat['p50_ms']:.0f}/{lat['p95_ms']:.0f}/{lat['p99_ms']:.0f} ms, "
      f"{lat['ms_per_frame']:.0f} ms/frame, RTF P95 {lat['rtf_p95']:.2f} (<= 1 meets the {lat['budget_ms']:.0f} ms budget)")
auc, decision = data["auc"], data["decision"]
fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
print(f"AUC frame {fmt(auc['frame'])}, window {fmt(auc['window'])}; AP frame {fmt(data['ap']['frame'])}; answer (p >= 0.5): "
      f"balanced accuracy {fmt(decision['balanced_accuracy'])}, recall {fmt(decision['recall'])}, "
      f"specificity {fmt(decision['specificity'])}")
for row in data["classification"]:
    if round(row["threshold"], 2) in (0.1, 0.3, 0.5, 0.7, 0.9):
        print(f"threshold {row['threshold']:.1f}: accuracy {row['accuracy']:.1%}, recall {row['recall']:.1%}, FNR {row['fnr']:.1%}")
print(f"full report: {sys.argv[1]}")
EOF
}

experiment() {
    local config="$1" out="$2"
    shift 2
    free -g | sed -n '1,2p'
    uv run s1stream run "$config" --out "$out" "$@"
    report "$out"
}

command="${1:-}"
shift || true
case "$command" in
    setup)  setup ;;
    assets) assets ;;
    smoke)  experiment "${1:-$DEFAULT_CONFIG}" "results/smoke/$(date +%Y%m%d-%H%M%S)" --limit-windows 5 ;;
    run)
        config="${1:-$DEFAULT_CONFIG}"
        experiment "$config" "results/$(sed -n 's/^name: *//p' "$config")/$(date +%Y%m%d-%H%M%S)" ;;
    report) report "${1:?usage: scripts/run.sh report <run dir>}" ;;
    all)    setup; assets; "$0" smoke "${1:-$DEFAULT_CONFIG}"; "$0" run "${1:-$DEFAULT_CONFIG}" ;;
    *)      sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
