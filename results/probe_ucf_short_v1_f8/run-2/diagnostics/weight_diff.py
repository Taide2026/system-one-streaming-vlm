"""Relative weight change ||C - base|| / ||base|| of d3-flash against Qwen3.5-9B, per tensor group."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
from safetensors import safe_open

def index(folder):
    out = {}
    for shard in sorted(Path(folder).glob("*.safetensors")):
        if shard.name == "readout.safetensors":
            continue
        with safe_open(str(shard), "pt") as f:
            for k in f.keys():
                out[k] = shard
    return out

def norm_name(k):
    k = re.sub(r"^(model\.|module\.)+", "", k)
    return k.replace("language_model.", "").replace("model.", "")

base, ft = index(sys.argv[1]), index(sys.argv[2])
bmap = {norm_name(k): k for k in base}
fmap = {norm_name(k): k for k in ft}
common = sorted(set(bmap) & set(fmap))
print("base tensors", len(base), "d3 tensors", len(ft), "matched", len(common))
print("only d3:", sorted(set(fmap) - set(bmap))[:10])
print("only base:", sorted(set(bmap) - set(fmap))[:10])
groups = defaultdict(lambda: [0.0, 0.0, 0])
for name in common:
    with safe_open(str(base[bmap[name]]), "pt") as fb, safe_open(str(ft[fmap[name]]), "pt") as ff:
        b = fb.get_tensor(bmap[name]).float(); c = ff.get_tensor(fmap[name]).float()
    if b.shape != c.shape:
        print("shape differs", name, b.shape, c.shape); continue
    m = re.match(r"(visual)\.|layers\.(\d+)\.", name)
    g = "visual" if name.startswith("visual") else (f"layer{int(m.group(2)):02d}" if m and m.group(2) else name)
    groups[g][0] += float(((c - b) ** 2).sum()); groups[g][1] += float((b ** 2).sum()); groups[g][2] += 1
res = {g: {"rel_change": (d / n) ** 0.5 if n else None, "tensors": t} for g, (d, n, t) in sorted(groups.items())}
for g, r in res.items():
    print(f"{g:40s} {r['rel_change']:.5f} ({r['tensors']} tensors)")
Path(sys.argv[3]).write_text(json.dumps(res, indent=1))
