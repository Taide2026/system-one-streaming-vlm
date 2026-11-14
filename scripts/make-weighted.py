#!/usr/bin/env python3
"""Derive scripts/train-floor-w.py from scripts/train-floor.py (2026-11-06): a per-record loss weight on the V1 task.
Config: v1_weight: {anomalous_negative: 3.0}  -> label-0 windows that come from anomalous training videos get that
weight in the soft cross-entropy (the KL anchor stays unweighted). Original file untouched."""
from pathlib import Path

src = Path("scripts/train-floor.py").read_text()
dst = Path("scripts/train-floor-w.py")
if dst.exists():
    raise SystemExit("exists")


def sub(old, new):
    global src
    assert src.count(old) == 1, (old[:60], src.count(old))
    src = src.replace(old, new)


# items carry (y, w) for v1
sub('''    items: list[tuple] = [("v1", r, None, float(r.y)) for r in train_records]
''', '''    wneg = float(data.get("v1_weight", {}).get("anomalous_negative", 1.0))
    def _w(r):
        return wneg if (float(r.y) < 0.5 and not str(r.category).lower().startswith("normal")) else 1.0
    items: list[tuple] = [("v1", r, None, (float(r.y), _w(r))) for r in train_records]
    print(f"v1 weights: {sum(1 for r in train_records if _w(r) != 1.0)} anomalous-video negatives x {wneg}", flush=True)
''')
# the loop: unpack (y, w) and weight the CE
sub('''            if task == "v1":
                y = torch.tensor(ys, device=logits.device, dtype=torch.float32)
                target = torch.stack([y, 1 - y], dim=1)''',
'''            wts = None
            if task == "v1":
                y = torch.tensor([v[0] if isinstance(v, (tuple, list)) else v for v in ys], device=logits.device, dtype=torch.float32)
                wts = torch.tensor([v[1] if isinstance(v, (tuple, list)) else 1.0 for v in ys], device=logits.device, dtype=torch.float32)
                target = torch.stack([y, 1 - y], dim=1)''')
sub('''            else:
                loss = soft_cross_entropy(logits, target)
                if anchor > 0:''',
'''            else:
                if wts is not None and float(wts.max()) != 1.0:
                    ce = -(target * torch.log_softmax(logits.float(), dim=-1)).sum(1)
                    loss = (wts * ce).sum() / wts.sum()
                else:
                    loss = soft_cross_entropy(logits, target)
                if anchor > 0:''')
# validation: ys are plain floats there already; the temperature code uses float(v) -> keep as is
dst.write_text(src)
print("wrote", dst)
