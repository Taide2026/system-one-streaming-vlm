"""Pick one state per condition/probe/feature by the validation score in fit.json (MIL: lowest video BCE; Holmes and
kNN: highest AUC), then report that state's test frame AUC with its bootstrap interval. Writes val_selected.json."""
import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
compare = json.loads((run / "preview-compare/compare.json").read_text())
lower_is_better = {"mil": True, "holmes": False, "knn": False}
out = {}
for c in "ABC":
    fit = json.loads((run / f"fit-{c}/fit.json").read_text())
    for m in fit["methods"]:
        for k in fit["kinds"]:
            val = np.array([fit["choices"][f"{m}/{k}/{s}"]["validation"] for s in range(fit["n_states"])], dtype=float)
            state = int(np.nanargmin(val) if lower_is_better[m] else np.nanargmax(val))
            e = compare["metrics"][f"{c}/{m}/{k}/frame_auc"]
            peak = int(np.nanargmax(e["value"]))
            out[f"{c}/{m}/{k}"] = {"val_state": state, "validation": float(val[state]),
                                   "test_frame_auc": e["value"][state], "low": e["low"][state], "high": e["high"][state],
                                   "test_peak_state": peak, "test_peak_auc": e["value"][peak]}
(run / "val_selected.json").write_text(json.dumps(out, indent=1))
for key, v in out.items():
    print(f"{key:18s} val@{v['val_state']:2d} test {v['test_frame_auc']:.3f} [{v['low']:.3f}, {v['high']:.3f}]   "
          f"(test-peak @{v['test_peak_state']:2d} {v['test_peak_auc']:.3f})")
