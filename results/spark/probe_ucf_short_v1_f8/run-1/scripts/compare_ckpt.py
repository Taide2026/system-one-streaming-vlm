"""Compare d3-flash and qwen3.5-9b checkpoints tensor by tensor (CPU, one tensor at a time)."""
import os
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open

A = Path(os.environ["S1_ASSETS"])
def index(d):
    m = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    return m
base, d3 = index(A / "qwen3.5-9b"), index(A / "d3-flash")
rename = lambda k: k.removeprefix("model.")
base_r = {rename(k): (k, f) for k, f in base.items() if not k.startswith(("lm_head.", "mtp."))}
d3k = set(d3)
print("d3 keys", len(d3k), "base backbone keys", len(base_r), "same key set", d3k == set(base_r))
print("only d3", sorted(d3k - set(base_r))[:5], "only base", sorted(set(base_r) - d3k)[:5])
print("base dropped", Counter(k.split(".")[0] for k in base if k.startswith(("lm_head.", "mtp."))))
handles = {}
def get(d, f, k):
    key = (d, f)
    if key not in handles:
        handles[key] = safe_open(str(d / f), "pt", device="cpu")
    return handles[key].get_tensor(k)
groups = {}
for k in sorted(d3k & set(base_r)):
    g = "visual" if k.startswith("visual.") else ("embed" if "embed_tokens" in k else ("norm" if k == "language_model.norm.weight" else "language_model.layers"))
    a = get(A / "d3-flash", d3[k], k); b = get(A / "qwen3.5-9b", base_r[k][1], base_r[k][0])
    same = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    s = groups.setdefault(g, [0, 0, 0])
    s[0] += 1; s[1] += same; s[2] += a.numel()
    if g == "language_model.layers" and same: print("same", k)
for g, (n, eq, numel) in groups.items():
    print(f"{g}: {n} tensors, {eq} identical, {numel/1e6:.1f} M params")
