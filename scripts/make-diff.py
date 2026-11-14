#!/usr/bin/env python3
"""Derive scripts/train-floor-diff.py from scripts/train-floor.py (2026-11-07): roadmap B2, the temporal-difference
input form at training time. Config: input_form: {name: diff, gain: 4} (train-floor only honours "foveated"; this
copy also honours "diff", applied to the v1 windows exactly where foveation would be). Original file untouched."""
from pathlib import Path

src = Path("scripts/train-floor.py").read_text()
dst = Path("scripts/train-floor-diff.py")
if dst.exists():
    raise SystemExit("exists")


def sub(old, new):
    global src
    assert src.count(old) == 1, (old[:60], src.count(old))
    src = src.replace(old, new)


sub('''    fov = fov if fov and fov.get("name") == "foveated" else None
''', '''    fov = fov if fov and fov.get("name") in ("foveated", "diff") else None
''')
sub('''        if task == "v1" and self.foveate_v1 is not None and images:
            from s1stream.input_forms.foveated import foveate

''', '''        if task == "v1" and self.foveate_v1 is not None and images and self.foveate_v1.get("name") == "diff":
            from s1stream.input_forms.diff import temporal_diff

            images = temporal_diff(images, float(self.foveate_v1.get("gain", 4.0)))
        elif task == "v1" and self.foveate_v1 is not None and images:
            from s1stream.input_forms.foveated import foveate

''')
dst.write_text(src)
print("wrote", dst)
