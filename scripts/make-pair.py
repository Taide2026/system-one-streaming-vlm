#!/usr/bin/env python3
"""Derive scripts/train-floor-pair.py from scripts/train-floor.py (2026-11-07): a within-video pairwise ranking task.
Config:
  tasks:
    v1pair: {n_pairs: 5000, margin: 0.3, weight: 1.0}
Each pair = (window inside an event, window outside any event) of the SAME anomalous training video, kept adjacent in
one batch; loss = soft CE on both (as v1) + weight * mean(relu(margin - (p_in - p_out))). The plain v1 task stays.
Original file untouched. Roadmap line A2."""
from pathlib import Path

src = Path("scripts/train-floor.py").read_text()
dst = Path("scripts/train-floor-pair.py")
if dst.exists():
    raise SystemExit("exists")


def sub(old, new):
    global src
    assert src.count(old) == 1, (old[:60], src.count(old))
    src = src.replace(old, new)


# encode: v1pair uses the v1 prompt table; all three floor classes define encode(), patch every one
_old = '''    def encode(self, task: str, record, images, payload) -> Any:
'''
assert src.count(_old) == 3, src.count(_old)
src = src.replace(_old, _old + '''        task = "v1" if task == "v1pair" else task
''')
# foveation applies to the pair windows too
sub('''        if task == "v1" and self.foveate_v1 is not None and images:''',
    '''        if task in ("v1", "v1pair") and self.foveate_v1 is not None and images:''')
# batches: keep pairs adjacent (no shuffle inside the v1pair group; chunks of `batch` = batch/2 pairs)
sub('''    by: dict[tuple, list[int]] = {}
    for i, it in enumerate(items):
        by.setdefault(key(it), []).append(i)
    rng = random.Random(seed)
    chunks = []
    for task, idx in by.items():
        rng.shuffle(idx)
        chunks += [idx[k:k + batch] for k in range(0, len(idx), batch)]
    rng.shuffle(chunks)
    return chunks''', '''    by: dict[tuple, list[int]] = {}
    for i, it in enumerate(items):
        by.setdefault(key(it), []).append(i)
    rng = random.Random(seed)
    chunks = []
    for task, idx in by.items():
        if task[0] == "v1pair":  # items are [in, out, in, out, ...]; shuffle pairs, keep each pair adjacent
            pairs = [idx[k:k + 2] for k in range(0, len(idx) - 1, 2)]
            rng.shuffle(pairs)
            flat = [i for p in pairs for i in p]
            chunks += [flat[k:k + batch] for k in range(0, len(flat), batch)]
            continue
        rng.shuffle(idx)
        chunks += [idx[k:k + batch] for k in range(0, len(idx), batch)]
    rng.shuffle(chunks)
    return chunks''')
# items: build the pairs
sub('''    items: list[tuple] = [("v1", r, None, float(r.y)) for r in train_records]
''', '''    items: list[tuple] = [("v1", r, None, float(r.y)) for r in train_records]
    pair_cfg = tasks.get("v1pair")
    if pair_cfg:
        from s1stream.traindata import windows_of as _wof  # noqa: F401  (same window cutter as sample_records)
        byv: dict = {}
        for r in train_records:
            if str(r.category).lower().startswith("normal"):
                continue
            byv.setdefault(r.video, {"in": [], "out": []})["in" if float(r.y) >= 0.5 else "out"].append(r)
        prng = random.Random(int(dspec.get("seed", 0)) + 23)
        vids = [v for v, d in byv.items() if d["in"] and d["out"]]
        n_pairs = int(pair_cfg.get("n_pairs", 5000)); made = 0
        while made < n_pairs and vids:
            v = prng.choice(vids); d = byv[v]
            a, b = prng.choice(d["in"]), prng.choice(d["out"])
            items.append(("v1pair", a, None, float(a.y))); items.append(("v1pair", b, None, float(b.y))); made += 1
        print(f"v1pair: {made} pairs from {len(vids)} anomalous videos with both kinds of window", flush=True)
''')
# loss: pair task = soft CE + margin ranking on p(Yes) of adjacent rows
sub('''            if task == "v1":
                y = torch.tensor(ys, device=logits.device, dtype=torch.float32)
                target = torch.stack([y, 1 - y], dim=1)''', '''            if task in ("v1", "v1pair"):
                y = torch.tensor(ys, device=logits.device, dtype=torch.float32)
                target = torch.stack([y, 1 - y], dim=1)''')
sub('''            else:
                loss = soft_cross_entropy(logits, target)
                if anchor > 0:''', '''            else:
                loss = soft_cross_entropy(logits, target)
                if task == "v1pair" and logits.shape[0] >= 2:
                    pc = data.get("tasks", {}).get("v1pair", {})
                    p_yes = torch.softmax(logits.float(), dim=-1)[:, 0]
                    n2 = (logits.shape[0] // 2) * 2
                    gap = p_yes[0:n2:2] - p_yes[1:n2:2]  # in - out, adjacent rows
                    loss = loss + float(pc.get("weight", 1.0)) * torch.relu(float(pc.get("margin", 0.3)) - gap).mean()
                if anchor > 0:''')
dst.write_text(src)
print("wrote", dst)
