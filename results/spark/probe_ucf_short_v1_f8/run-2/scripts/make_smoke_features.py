"""Copy one state (index given) of a features run into a smaller features run for a probe-fit smoke test."""
import shutil
import sys
from pathlib import Path

import numpy as np

source, target, state = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
for name in ("train_anomaly", "train_normal", "test"):
    (target / name).mkdir(parents=True, exist_ok=False)
    for kind in ("last", "visual"):
        np.save(target / name / f"{kind}.npy", np.load(source / name / f"{kind}.npy", mmap_mode="r")[:, state:state + 1])
    shutil.copy(source / name / "meta.jsonl", target / name / "meta.jsonl")
print(target)
