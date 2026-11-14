#!/usr/bin/env python3
"""Derive scripts/train-floor-joint.py from scripts/train-floor.py (2026-11-05): a second data source (data_xd) whose
windows are trained as task "v1_xd" with the XD V1 prompt, soft coverage target, same loss and KL anchor. The
original file is left untouched. Idempotent: refuses to run if the target already exists."""
from pathlib import Path

src = Path("scripts/train-floor.py").read_text()
dst = Path("scripts/train-floor-joint.py")
if dst.exists():
    raise SystemExit("exists")

def sub(old, new, count=1):
    global src
    assert src.count(old) == count, (old[:60], src.count(old))
    src = src.replace(old, new)

# a) D3Floor: the v1_xd table (XD V1 prompt rendered through the release's own question builder)
sub('''        self.states = {t: PROMPTS[t]["state"] for t in self.tables}
        self.readout0 = self.base.readout.detach().clone()''',
'''        qxd = question_of({"question": {"prompt": "v1_xd", **PROMPTS["v1_xd"]}})
        self.tables["v1_xd"] = (self.code.normalize_question(build_questions(qxd)["answer"]).rendered, tuple(qxd.options))
        self.states = {t: PROMPTS[t]["state"] for t in self.tables}
        self.readout0 = self.base.readout.detach().clone()''')

# b) main: read the second HIVAU source and sample its windows
sub('''    write_records(train_records, out_dir / "records.jsonl")
    write_records(val_records, out_dir / "val.jsonl")
''',
'''    write_records(train_records, out_dir / "records.jsonl")
    write_records(val_records, out_dir / "val.jsonl")
    xd_records: list = []
    xd = data.get("data_xd")
    if xd:
        items_xd = read_hivau(Path(os.path.expandvars(os.path.expanduser(xd["hivau_json"]))), resolve(xd["asset"]),
                              str(xd.get("videos_dir", "videos")), bool(xd.get("flat", False)),
                              bool(xd.get("skip_missing", False)))
        n_xd = int(tasks.get("v1_xd", {}).get("n", 10000))
        xd_records, _ = sample_records(items_xd, float(window["length_s"]), float(window["stride_s"]),
                                       int(window["frames"]), n_xd // 2, n_xd - n_xd // 2,
                                       int(xd.get("n_val", 2000)), float(xd.get("holdout", 0.1)),
                                       int(xd.get("seed", 0)))
        write_records(xd_records, out_dir / "records-xd.jsonl")
        print(f"v1_xd: {len(xd_records)} records", flush=True)
''')

sub('''    items: list[tuple] = [("v1", r, None, float(r.y)) for r in train_records]
''',
'''    items: list[tuple] = [("v1", r, None, float(r.y)) for r in train_records]
    items.extend(("v1_xd", r, None, float(r.y)) for r in xd_records)
''')

# c) the soft two-option target applies to both V1 tasks
sub('''            if task == "v1":
                y = torch.tensor(ys, device=logits.device, dtype=torch.float32)''',
'''            if task in ("v1", "v1_xd"):
                y = torch.tensor(ys, device=logits.device, dtype=torch.float32)''')

dst.write_text(src)
print("wrote", dst)
