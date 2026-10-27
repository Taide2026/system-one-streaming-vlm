"""Recompute P(Yes) from state 32 of last.npy with code tokens A/B and with words Yes/No (CPU)."""
import os
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

A = Path(os.environ["S1_ASSETS"])
root = Path(os.environ["S1_FEATURES"] + "/probe_ucf_short_v1_f8")
tok = json.loads((A / "qwen3.5-9b/tokenizer.json").read_text())["model"]["vocab"]
ids = {"A": tok["A"], "B": tok["B"], "Yes": tok["Yes"], "No": tok["No"]}
index = json.loads((A / "qwen3.5-9b/model.safetensors.index.json").read_text())["weight_map"]
with safe_open(str(A / "qwen3.5-9b" / index["lm_head.weight"]), "pt", device="cpu") as f:
    sl = f.get_slice("lm_head.weight")
    head = {k: sl[i:i + 1][0].to(torch.bfloat16) for k, i in ids.items()}
with safe_open(str(A / "d3-flash/readout.safetensors"), "pt", device="cpu") as f:
    readout = f.get_tensor("weight")[:2]


def two_way(h, a, b):
    return torch.softmax(torch.stack((h @ a, h @ b), dim=-1).float(), dim=-1)[:, 0].numpy()


out = {}
for c in "CAB":
    for s in ("train_anomaly", "train_normal", "test"):
        h = torch.from_numpy(np.asarray(np.load(root / c / "run-1" / s / "last.npy", mmap_mode="r")[:, 32], dtype=np.float32))
        stored = np.array([json.loads(l)["p_yes"] for l in open(root / c / "run-1" / s / "meta.jsonl")])
        hb = h.to(torch.bfloat16)
        row = {"stored": stored,
               "head_AB": two_way(hb, head["A"], head["B"]),
               "head_YesNo": two_way(hb, head["Yes"], head["No"]),
               "readout_AB": two_way(h, readout[0], readout[1])}
        out[(c, s)] = row
        print(c, s, {k: round(float(v.mean()), 3) for k, v in row.items()},
              "stored vs recomputed max diff", round(float(np.abs(stored - (row["readout_AB"] if c == "C" else row["head_YesNo"])).max()), 4))
np.save(Path(__file__).with_name("pyes_recompute.npy"), {f"{c}/{s}/{k}": v for (c, s), r in out.items() for k, v in r.items()}, allow_pickle=True)
