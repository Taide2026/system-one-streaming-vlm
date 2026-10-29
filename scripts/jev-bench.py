#!/usr/bin/env python3
"""jev-bench (Praveenrajus/jev-bench) on a d3 checkpoint, zero-shot or with a fine-tuning run merged (2026-10-29):

  HF_HOME=... uv run --frozen --with huggingface_hub python scripts/jev-bench.py --asset d3-nano [--adapter <run dir>]
      --out <results dir> [--configs boolq,banking77,...] [--limit N]

Each record is a System One request already ({state, question: {type, instructions, criteria...}, label}); the d3
runtime answers it in one pass (runtime.system_one). Metrics per config: choice -> accuracy of the argmax and 10-bin
ECE of the chosen option's probability; noul -> accuracy at 0.5 and ECE of p(yes); score -> accuracy of the argmax
level and the mean absolute level error. Writes <out>/<config>.jsonl and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import s1stream.plugins  # noqa: E402,F401
from s1stream.assets import resolve  # noqa: E402
from s1stream.registry import BACKBONES  # noqa: E402

REPO = "Praveenrajus/jev-bench"


def ece(p: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1); tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += abs(correct[m].mean() - p[m].mean()) * m.mean()
    return float(tot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="d3-nano"); ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--configs", default=None)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--split", default="test")
    args = ap.parse_args()
    spec = {"name": "d3", "asset": args.asset, "path": str(resolve(args.asset)), "memory_cap_gb": 40}
    if args.adapter:
        spec["adapter"] = args.adapter
    runtime = BACKBONES.build(spec).model
    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith("data/") and f.endswith(f"/{args.split}.jsonl"))
    configs = [f.split("/")[1] for f in files]
    if args.configs:
        want = args.configs.split(","); configs = [c for c in configs if c in want]
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cfg in configs:
        if (args.out / "summary.json").is_file() and cfg in json.loads((args.out / "summary.json").read_text()):
            summary = json.loads((args.out / "summary.json").read_text()); summary.pop("_mean", None)
            print(f"{cfg:26s} done already, skipped", flush=True)
            continue
        path = hf_hub_download(REPO, f"data/{cfg}/{args.split}.jsonl", repo_type="dataset")
        rows, bad = [], 0
        for l in Path(path).read_text(errors="replace").splitlines():
            if not l.strip():
                continue
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                bad += 1
        if bad:
            print(f"{cfg}: {bad} unparsable lines skipped", flush=True)
        if not rows:
            print(f"{cfg}: no records, skipped", flush=True)
            continue
        if args.limit:
            rows = rows[:args.limit]
        kind = rows[0]["primitive"]
        probs, correct, errs, n_err, t0 = [], [], [], 0, time.perf_counter()
        with (args.out / f"{cfg}.jsonl").open("w") as f:
            for r in rows:
                state = json.loads(r["state"]) if isinstance(r["state"], str) and r["state"][:1] in "{[" else r["state"]
                question = json.loads(r["question"]) if isinstance(r["question"], str) else r["question"]
                ans = runtime.system_one(state=state, questions={"q": question})["answers"]["q"]
                if "error" in ans:
                    n_err += 1
                    f.write(json.dumps({"id": r["id"], "error": ans}) + "\n")
                    continue
                label = str(r["label"])
                if kind == "noul":
                    p = float(ans["noul"]); pred = "1" if p >= 0.5 else "0"
                    ok = pred == label; probs.append(p if pred == "1" else 1 - p); correct.append(ok)
                elif kind == "choice":
                    pred = ans["choice"]; ok = pred == label
                    probs.append(float(ans["probabilities"][pred])); correct.append(ok)
                else:  # score: {"score": expected level, "probabilities": {level key: p}}; the prediction is the argmax level
                    keys = list(ans["probabilities"]); pred = max(keys, key=lambda k: ans["probabilities"][k]); ok = pred == label
                    probs.append(float(ans["probabilities"][pred])); correct.append(ok)
                    if label in keys:
                        errs.append(abs(keys.index(pred) - keys.index(label)))
                f.write(json.dumps({"id": r["id"], "label": label, "pred": pred, "p": probs[-1], "correct": bool(ok)}) + "\n")
        c = np.array(correct, float); p = np.array(probs, float)
        summary[cfg] = {"primitive": kind, "n": int(len(rows)), "errors": n_err, "bad_lines": bad, "accuracy": 100 * float(c.mean()) if len(c) else None,
                        "ece": ece(p, c) if len(c) else None, "mae_levels": float(np.mean(errs)) if errs else None,
                        "s_per_item": (time.perf_counter() - t0) / max(len(rows), 1)}
        print(f"{cfg:26s} {kind:6s} n {len(rows):5d}  acc {summary[cfg]['accuracy']:5.1f}  ECE {summary[cfg]['ece']:.3f}"
              f"  err {n_err}  {summary[cfg]['s_per_item']:.2f} s/item", flush=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    accs = [v["accuracy"] for v in summary.values() if v["accuracy"] is not None]
    eces = [v["ece"] for v in summary.values() if v["ece"] is not None]
    summary["_mean"] = {"accuracy": float(np.mean(accs)), "ece": float(np.mean(eces)), "configs": len(accs)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"mean over {len(accs)} configs: accuracy {np.mean(accs):.1f}, ECE {np.mean(eces):.3f}")


if __name__ == "__main__":
    main()
