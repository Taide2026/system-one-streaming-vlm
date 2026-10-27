"""Refit Holmes / last heads with the original Adam setting (lr 1e-2, 200 steps) and with lr 1e-3 for 1000 steps;
report the loss trace and train / test window AUC. Writes diagnostics/holmes_lr.json."""
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from s1stream.probe import fit as F

run = Path(sys.argv[1])
root = Path(os.environ["S1_FEATURES"] + "/probe_ucf_short_v1_f8")
out = {}
for cond, state in [("C", 24), ("C", 27), ("C", 31), ("C", 23), ("B", 31)]:
    feats = root / cond / "run-1"
    meta = [json.loads(l) for s in ("train_anomaly", "train_normal") for l in open(feats / s / "meta.jsonl")]
    y_test = np.array([json.loads(l)["label"] for l in open(feats / "test" / "meta.jsonl")])
    train = np.concatenate([np.asarray(np.load(feats / s / "last.npy", mmap_mode="r")[:, state], dtype=np.float32)
                            for s in ("train_anomaly", "train_normal")])
    test = np.asarray(np.load(feats / "test" / "last.npy", mmap_mode="r")[:, state], dtype=np.float32)
    x, tx = F.standardize(train, test)
    y = torch.tensor([r["holmes"] for r in meta], dtype=torch.float32)
    lam = json.loads((run / f"fit-{cond}/fit.json").read_text())["choices"][f"holmes/last/{state}"]["setting"]["lam"]
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - float(y.mean())) / float(y.mean())))
    for lr, steps in [(1e-2, 200), (1e-3, 1000)]:
        torch.manual_seed(F.SEED)
        head = torch.nn.Linear(x.shape[1], 1)
        torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        losses = []
        for _ in range(steps):
            opt.zero_grad()
            loss = bce(head(x).squeeze(1), y) + lam * head.weight.square().sum()
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        with torch.no_grad():
            tr, te = head(x).squeeze(1).numpy(), head(tx).squeeze(1).numpy()
        ups = int(sum(b > a * 1.05 for a, b in zip(losses, losses[1:])))
        key = f"{cond}/{state}/lr{lr:g}x{steps}"
        out[key] = {"lam": lam, "final_loss": losses[-1], "min_loss": min(losses), "steps_where_loss_rose_5pct": ups,
                    "train_auc": float(roc_auc_score(y.numpy(), tr)), "test_window_auc": float(roc_auc_score(y_test, te))}
        print(key, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[key].items()})
        sys.stdout.flush()
(run / "diagnostics/holmes_lr.json").write_text(json.dumps(out, indent=1))
