#!/usr/bin/env python3
"""Derive scripts/train-floor-dec.py from scripts/train-floor.py (2026-11-07): roadmap A3, the decoupled read-out.
Config:
  tasks:
    v1:  {n: 10000}                      # head V: p(video anomalous); target = the video-level label (video_target: video)
    v1m: {video_target: video}           # head M: p(moment | video); target = 0.5 + coverage - mean coverage of the
                                         # video's sampled windows, clipped to [0, 1]; anomalous training videos only
Two read-out heads share the LoRA. Head V is the ordinary readout (saved as usual); head M is saved under
<run>/moment/ (adapter symlink + its own readout.safetensors), so both can be evaluated with backbone.adapter.
Original file untouched."""
from pathlib import Path

src = Path("scripts/train-floor.py").read_text()
dst = Path("scripts/train-floor-dec.py")
if dst.exists():
    raise SystemExit("exists")


def sub(old, new):
    global src
    assert src.count(old) == 1, (old[:60], src.count(old))
    src = src.replace(old, new)


_old = '''    def encode(self, task: str, record, images, payload) -> Any:
'''
assert src.count(_old) == 3, src.count(_old)
src = src.replace(_old, _old + '''        task = "v1" if task == "v1m" else task
''')
sub('''        if task == "v1" and self.foveate_v1 is not None and images:''',
    '''        if task in ("v1", "v1m") and self.foveate_v1 is not None and images:''')
# D3Floor: the second head
sub('''        self.readout0 = self.base.readout.detach().clone()
        self.option_ids = self.base.option_ids  # v1, for validation
''', '''        self.readout0 = self.base.readout.detach().clone()
        self.option_ids = self.base.option_ids  # v1, for validation
        self.readout_m = torch.nn.Parameter(self.base.readout.detach().clone())  # head M: p(moment | video)
        self.active_head = "v"
''')
sub('''        readout = self.readout0.to(hidden.device) if reference else self.base.readout
        batch["_n_options"] = n
        return (hidden.float() @ readout.T)[:, :n]''',
    '''        readout = self.readout0.to(hidden.device) if reference else (self.readout_m if self.active_head == "m" else self.base.readout)
        batch["_n_options"] = n
        return (hidden.float() @ readout.T)[:, :n]''')
sub('''        lora, _ = self.base.parameters()
        return lora, [self.base.readout]
''', '''        lora, _ = self.base.parameters()
        return lora, [self.base.readout, self.readout_m]
''')
# items
sub('''    items: list[tuple] = [("v1", r, None, float(r.y)) for r in train_records]
''', '''    dec = tasks.get("v1m")
    if dec is not None:
        if not hasattr(floor, "readout_m"):
            raise ConfigError("v1m (decoupled read-out) is implemented for the d3 floor only")
        is_anom = lambda r: not str(r.category).lower().startswith("normal")  # noqa: E731
        video_target = str(dec.get("video_target", "video"))
        items: list[tuple] = [("v1", r, None, ((1.0 if is_anom(r) else 0.0) if video_target == "video" else float(r.y)))
                              for r in train_records]
        byv: dict = {}
        for r in train_records:
            if is_anom(r):
                byv.setdefault(r.video, []).append(r)
        for rs in byv.values():
            m = sum(float(r.y) for r in rs) / len(rs)
            for r in rs:
                items.append(("v1m", r, None, min(1.0, max(0.0, 0.5 + float(r.y) - m))))
        n_m = sum(1 for it in items if it[0] == "v1m")
        print(f"decoupled: head V target = {video_target}; head M: {n_m} windows of {len(byv)} anomalous videos", flush=True)
    else:
        items = [("v1", r, None, float(r.y)) for r in train_records]
''')
# loop: pick the head; loss as v1
sub('''        for batch, task, ys in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = floor.forward(batch)
            n_opt = logits.shape[1]
            if task == "v1":''', '''        for batch, task, ys in loader:
            if hasattr(floor, "active_head"):
                floor.active_head = "m" if task == "v1m" else "v"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = floor.forward(batch)
            n_opt = logits.shape[1]
            if task in ("v1", "v1m"):''')
# validation with head V
sub('''    # ---- temperature on the held-out v1 windows
    floor.train(False)
''', '''    # ---- temperature on the held-out v1 windows
    floor.train(False)
    if hasattr(floor, "active_head"):
        floor.active_head = "v"
''')
sub('''    floor.save(out_dir)
''', '''    floor.save(out_dir)
    if hasattr(floor, "readout_m") and tasks.get("v1m") is not None:
        from safetensors.torch import save_file

        md = out_dir / "moment"; md.mkdir(exist_ok=True)
        if not (md / "adapter").exists():
            (md / "adapter").symlink_to("../adapter")
        save_file({"weight": floor.readout_m.detach().cpu().contiguous()}, md / "readout.safetensors")
        (md / "calibration.json").write_text((out_dir / "calibration.json").read_text())  # same T; ranks unaffected
        print(f"head M -> {md}", flush=True)
''')
dst.write_text(src)
print("wrote", dst)
