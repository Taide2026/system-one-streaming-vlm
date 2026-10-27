"""Time one Holmes-style head fit (Adam, full batch) on CPU and CUDA for C / last / state 31, and one kNN scoring.
Writes diagnostics/bench_gpu_fit.json."""
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from s1stream.probe import fit as F

out_dir = Path(sys.argv[1])
total = torch.cuda.get_device_properties(0).total_memory / 2**30
torch.cuda.set_per_process_memory_fraction(8 / total)
feats = Path(os.environ["S1_FEATURES"] + "/probe_ucf_short_v1_f8/C/run-1")
train = np.concatenate([np.asarray(np.load(feats / s / "last.npy", mmap_mode="r")[:, 31], dtype=np.float32)
                        for s in ("train_anomaly", "train_normal")])
meta = [json.loads(l) for s in ("train_anomaly", "train_normal") for l in open(feats / s / "meta.jsonl")]
normal = np.array([r["category"] == "Normal" for r in meta])
x, = F.standardize(train)
y = torch.tensor([r["holmes"] for r in meta], dtype=torch.float32)
result = {}
for device, lr, steps in [("cpu", 1e-2, 200), ("cpu", 1e-3, 1000), ("cuda", 1e-2, 200), ("cuda", 1e-3, 1000)]:
    xd, yd = x.to(device), y.to(device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - float(y.mean())) / float(y.mean()), device=device))
    torch.manual_seed(F.SEED)
    head = torch.nn.Linear(x.shape[1], 1).to(device)
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        loss = bce(head(xd).squeeze(1), yd) + 1e-3 * head.weight.square().sum()
        loss.backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    result[f"{device}/lr{lr:g}x{steps}"] = round(time.perf_counter() - started, 3)
started = time.perf_counter()
F.knn_scores(train[normal], train[:1684])
result["knn_cpu_one_call"] = round(time.perf_counter() - started, 3)
result["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
print(result)
(out_dir / "diagnostics/bench_gpu_fit.json").write_text(json.dumps(result, indent=1))
