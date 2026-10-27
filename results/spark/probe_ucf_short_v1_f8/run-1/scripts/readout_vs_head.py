"""Compare d3-flash readout rows with the base lm_head rows of the code tokens (CPU, slices only)."""
import os
import json
from pathlib import Path

import torch
from safetensors import safe_open

A = Path(os.environ["S1_ASSETS"])
vocab = json.loads((A / "qwen3.5-9b/tokenizer.json").read_text())["model"]["vocab"]
with safe_open(str(A / "d3-flash/readout.safetensors"), "pt", device="cpu") as f:
    print({k: (f.get_slice(k).get_shape(), f.get_slice(k).get_dtype()) for k in f.keys()})
    readout = f.get_tensor(list(f.keys())[0]).float()
index = json.loads((A / "qwen3.5-9b/model.safetensors.index.json").read_text())["weight_map"]
head_file = A / "qwen3.5-9b" / index["lm_head.weight"]
ids = [vocab[c] for c in "ABCDE"]
with safe_open(str(head_file), "pt", device="cpu") as f:
    head = torch.stack([f.get_slice("lm_head.weight")[i:i + 1][0] for i in ids]).float()
cos = torch.nn.functional.cosine_similarity
for r, c in enumerate("ABCDE"):
    print(c, "id", ids[r], "cos(readout row, base lm_head row)", round(cos(readout[r], head[r], dim=0).item(), 4),
          "max cos to other code rows", round(max(cos(readout[r], head[j], dim=0).item() for j in range(5) if j != r), 4),
          "norm readout", round(readout[r].norm().item(), 2), "norm head", round(head[r].norm().item(), 2))
import string
letters = string.ascii_uppercase
ids = [vocab[c] for c in letters]
with safe_open(str(head_file), "pt", device="cpu") as f:
    sl = f.get_slice("lm_head.weight")
    head = torch.stack([sl[i:i + 1][0] for i in ids]).float()
print("A-Z exact equal:", torch.equal(readout[:26], head), "max abs diff", (readout[:26] - head).abs().max().item())
with safe_open(str(A / "d3-flash/readout.safetensors"), "pt", device="cpu") as f:
    pass
print("rows 26.. norms", [round(x, 2) for x in readout[26:30].norm(dim=1).tolist()])
r = readout[:26]
print("readout representable in bf16:", torch.equal(r.to(torch.bfloat16).float(), r))
d = (r - head).abs()
print("fraction of elements that differ", (d > 0).float().mean().item(), "rel diff norm", ((r - head).norm() / head.norm()).item())
print("diff in bf16 ulps: max", (d / (head.abs().clamp_min(1e-8) * 2**-8)).max().item())
