"""MLP probe vs linear probe on selected feature states (2026-10-27 derived experiment of vit-1).

Question: does a nonlinear read-out find anomaly information that the linear probe misses?
Everything except the head is as probe-fit (s1stream.probe.fit): standardized features, the same validation split,
MIL with top-k pooling (KS) on video labels, the Holmes window head with pos_weight, full-batch Adam, STEPS steps.
Head: Linear(d, HIDDEN) -> GELU -> Dropout(DROPOUT) -> Linear(HIDDEN, 1). Grid: weight decay in WDS (AdamW);
MIL also k in KS. Choice on validation (MIL: lowest video BCE; Holmes: highest window AUC), refit on all train windows,
score the test windows. Runs on cuda with a memory cap of CAP_GB (resource rule); peak GPU memory is printed.

Usage (worktree root): uv run python results/spark/probe_ucf_short_v1_f8/mlp-1/scripts/mlp_probe.py <name>=<features dir>:<kind>:<state> ...
  e.g. merger=$S1_FEATURES/probe_ucf_short_v1_f8/M/vit-1:visual:0
Writes ../targets/<name>.json (choices, losses) and ../targets/<name>.npy (scores (2, n_test): mil, holmes).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from s1stream.probe.fit import KS, SEED, STEPS, bags, split, standardize, topk_pool

HIDDEN, DROPOUT, LR, CAP_GB = 256, 0.1, 1e-3, 4.0
DEVICE = "cuda"
WDS = (1e-4, 1e-2, 1e-1)
OUT = Path(__file__).resolve().parents[1] / "targets"


def head(dim: int) -> torch.nn.Module:
    torch.manual_seed(SEED)
    return torch.nn.Sequential(torch.nn.Linear(dim, HIDDEN), torch.nn.GELU(), torch.nn.Dropout(DROPOUT),
                               torch.nn.Linear(HIDDEN, 1)).to(DEVICE)


def train(x: torch.Tensor, loss_fn, wd: float) -> tuple[torch.nn.Module, float]:
    model = head(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)
    model.train()
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = loss_fn(model(x).squeeze(1))
        loss.backward()
        optimizer.step()
    model.eval()
    return model, float(loss.detach())


def score(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(x).squeeze(1).cpu()


def run(name: str, features: Path, kind: str, state: int) -> None:
    started = time.monotonic()
    meta = {s: [json.loads(line) for line in (features / s / "meta.jsonl").read_text().splitlines()]
            for s in ("train_anomaly", "train_normal", "test")}
    train_meta = meta["train_anomaly"] + meta["train_normal"]
    videos = [row["video"] for row in train_meta]
    video_y = {row["video"]: float(row["category"] != "Normal") for row in train_meta}
    is_val = np.array([v in split(videos, {row["video"]: row["category"] for row in train_meta}) for v in videos])
    holmes = torch.tensor([row["holmes"] for row in train_meta], dtype=torch.float32)
    arrays = {s: np.load(features / s / f"{kind}.npy", mmap_mode="r") for s in meta}
    x_train = np.concatenate([np.asarray(arrays[s][:, state], dtype=np.float32)
                              for s in ("train_anomaly", "train_normal")])
    x_test = np.asarray(arrays["test"][:, state], dtype=np.float32)

    def bag_part(selected):
        index, mask, order = bags([v for v, keep in zip(videos, selected) if keep])
        return index, mask, torch.tensor([video_y[n] for n in order])

    fit_bag, val_bag, all_bag = bag_part(~is_val), bag_part(is_val), bag_part(np.ones(len(videos), dtype=bool))
    bce = torch.nn.BCEWithLogitsLoss()
    fit_x, val_x = (t.to(DEVICE) for t in standardize(x_train[~is_val], x_train[is_val]))
    holmes_d = holmes.to(DEVICE)
    fit_bag, all_bag = (tuple(t.to(DEVICE) for t in bag) for bag in (fit_bag, all_bag))

    def mil_loss(bag, k):
        return lambda s: bce(topk_pool(s, bag[0], bag[1], k), bag[2])

    def holmes_loss(y):
        pos = y.mean().clamp(min=1e-6)
        weighted = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)
        return lambda s: weighted(s, y)

    best_mil = (float("inf"), None, None)
    for k in KS:
        for wd in WDS:
            model, _ = train(fit_x, mil_loss(fit_bag, k), wd)
            value = float(bce(topk_pool(score(model, val_x), val_bag[0], val_bag[1], k), val_bag[2]))
            if value < best_mil[0]:
                best_mil = (value, k, wd)
    best_holmes = (-float("inf"), None)
    for wd in WDS:
        model, _ = train(fit_x, holmes_loss(holmes_d[torch.from_numpy(~is_val).to(DEVICE)]), wd)
        value = float(roc_auc_score(holmes[is_val].numpy(), score(model, val_x).numpy()))
        if value > best_holmes[0]:
            best_holmes = (value, wd)

    all_x, test_x = (t.to(DEVICE) for t in standardize(x_train, x_test))
    mil_model, mil_final = train(all_x, mil_loss(all_bag, best_mil[1]), best_mil[2])
    holmes_model, holmes_final = train(all_x, holmes_loss(holmes_d), best_holmes[1])
    scores = np.stack([score(mil_model, test_x).numpy(), score(holmes_model, test_x).numpy()]).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{name}.npy", scores)
    (OUT / f"{name}.json").write_text(json.dumps({
        "features": str(features), "kind": kind, "state": state, "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
        "steps": STEPS, "mil": {"k": best_mil[1], "wd": best_mil[2], "validation_bce": best_mil[0],
                                "final_loss": mil_final},
        "holmes": {"wd": best_holmes[1], "validation_auc": best_holmes[0], "final_loss": holmes_final},
        "seconds": time.monotonic() - started}, indent=1))
    print(f"{name}: mil k={best_mil[1]} wd={best_mil[2]} val bce {best_mil[0]:.4f}; holmes wd={best_holmes[1]} "
          f"val auc {best_holmes[0]:.4f}; {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(CAP_GB / total)
    for spec in sys.argv[1:]:
        name, rest = spec.split("=", 1)
        path, kind, state = rest.rsplit(":", 2)
        if (OUT / f"{name}.npy").exists():
            print(f"{name}: exists, skipped")
            continue
        run(name, Path(path), kind, int(state))
    print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
