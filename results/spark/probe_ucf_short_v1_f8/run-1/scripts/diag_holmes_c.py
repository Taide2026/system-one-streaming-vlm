"""Diagnose the Holmes / last dips of condition C: feature statistics, refit with the chosen lambda, training loss,
and how the test scores are formed. Writes diagnostics/holmes_c.json."""
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
cases = [("C", 23), ("C", 24), ("C", 27), ("C", 31), ("C", 32), ("B", 31)]
out = {}
for cond, state in cases:
    feats = root / cond / "run-1"
    meta = {s: [json.loads(l) for l in open(feats / s / "meta.jsonl")] for s in ("train_anomaly", "train_normal", "test")}
    train = np.concatenate([np.asarray(np.load(feats / s / "last.npy", mmap_mode="r")[:, state], dtype=np.float32)
                            for s in ("train_anomaly", "train_normal")])
    test = np.asarray(np.load(feats / "test" / "last.npy", mmap_mode="r")[:, state], dtype=np.float32)
    holmes = torch.tensor([r["holmes"] for r in meta["train_anomaly"] + meta["train_normal"]], dtype=torch.float32)
    y_test = np.array([r["label"] for r in meta["test"]])
    std = train.std(axis=0)
    all_x, test_x = F.standardize(train, test)
    lam = json.loads((run / f"fit-{cond}/fit.json").read_text())["choices"][f"holmes/last/{state}"]["setting"]["lam"]
    # refit with a loss trace (same procedure as F.fit_window / F._fit_linear)
    pos = float(holmes.mean())
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
    torch.manual_seed(F.SEED)
    head = torch.nn.Linear(all_x.shape[1], 1)
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    opt = torch.optim.Adam(head.parameters(), lr=F.LR)
    trace = []
    for step in range(F.STEPS):
        opt.zero_grad()
        loss = bce(head(all_x).squeeze(1), holmes) + lam * head.weight.square().sum()
        loss.backward(); opt.step()
        if step in (0, 9, 49, 99, 149, 199):
            trace.append(round(float(loss), 4))
    with torch.no_grad():
        tr = head(all_x).squeeze(1).numpy(); te = head(test_x).squeeze(1).numpy()
    saved = np.load(run / f"fit-{cond}/test_scores.npy")[1, 0, state]
    w = head.weight.detach().numpy()[0]
    contrib = np.abs(test_x.numpy() * w)  # per-window, per-dim contribution
    top_dims = np.argsort(-np.abs(w))[:5]
    out[f"{cond}/{state}"] = {
        "lam": lam, "train_absmax": float(np.abs(train).max()), "test_absmax": float(np.abs(test).max()),
        "dims_std_below_1e-2": int((std < 1e-2).sum()),
        "test_z_absmax": float(np.abs(test_x.numpy()).max()), "train_z_absmax": float(np.abs(all_x.numpy()).max()),
        "loss_trace_steps_0_9_49_99_149_199": trace,
        "train_holmes_auc": float(roc_auc_score(holmes.numpy(), tr)),
        "test_window_auc": float(roc_auc_score(y_test, te)),
        "refit_equals_saved_maxdiff": float(np.abs(te - saved).max()),
        "weight_norm": float(np.linalg.norm(w)), "bias": float(head.bias),
        "test_score_range": [float(te.min()), float(te.max())], "train_score_range": [float(tr.min()), float(tr.max())],
        "top_weight_dims": [int(d) for d in top_dims],
        "max_single_dim_share_of_test_score": float((contrib.max(axis=1) / contrib.sum(axis=1)).max()),
    }
    print(cond, state, json.dumps(out[f"{cond}/{state}"]))
    sys.stdout.flush()
(run / "diagnostics/holmes_c.json").write_text(json.dumps(out, indent=1))
