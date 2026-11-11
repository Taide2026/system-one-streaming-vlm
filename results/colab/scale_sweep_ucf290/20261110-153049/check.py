# Run on the Colab VM: print DONE_OK when run-6.sh has ended and all 6 score files on Drive have 18,390 lines.
import pathlib, subprocess
log = pathlib.Path("/content/run-6.log").read_text() if pathlib.Path("/content/run-6.log").exists() else ""
jobs = [(m, u) for u in ("bicubic3", "bicubic1.75", "bicubic1.5") for m in ("fov2", "mixkl")]
counts = {}
for m, u in jobs:
    p = pathlib.Path(f"/content/drive/MyDrive/s1-colab-scale-score/out/{m}/{u}/scores.jsonl")
    counts[f"{m}/{u}"] = sum(1 for _ in p.open()) if p.exists() else 0
print("ALL_DONE" in log, "FAILED" in log, counts)
if "ALL_DONE" in log and all(c == 18390 for c in counts.values()):
    print("DONE_OK")
