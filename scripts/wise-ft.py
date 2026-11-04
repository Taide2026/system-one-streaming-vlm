#!/usr/bin/env python3
"""WiSE-FT baseline (2026-11-01): weight-space interpolation between the release and a fine-tuned run, the
one-line alternative a reviewer will ask about - can alpha-scaling today's LoRA (arm A) match the floor on both
axes (UCF-290 AND the generalization suite) without any multi-task training?

  python scripts/wise-ft.py --run <ft run dir> --asset d3-nano --alpha 0.5 --out <derived run dir>

LoRA makes the interpolation exact and free: W = W0 + alpha * B A, so scaling every lora_B tensor by alpha IS
WiSE-FT on the backbone. The d3 read-out and the temperature are interpolated linearly with the release's. The
output folder is a normal run dir (adapter/ + readout.safetensors + calibration.json) that any eval config's
backbone.adapter can point at."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1stream.assets import resolve  # noqa: E402


def main() -> None:
    from safetensors.torch import load_file, save_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="fine-tuned run dir (adapter/ + readout.safetensors)")
    ap.add_argument("--asset", default="d3-nano")
    ap.add_argument("--alpha", type=float, required=True, help="0 = release (or --run2), 1 = the fine-tuned run")
    ap.add_argument("--run2", type=Path, default=None,
                    help="interpolate toward this run instead of the release (both runs must share the LoRA init, "
                    "e.g. the same data seed; used for the fov2<->fov2h small/medium dial, 2026-11-04)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    a = args.alpha
    args.out.mkdir(parents=True, exist_ok=True)

    src = args.run / "adapter"
    (args.out / "adapter").mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.suffix == ".safetensors":
            t = load_file(f)
            if args.run2 is not None:  # elementwise blend of two adapters sharing the init
                t2 = load_file(args.run2 / "adapter" / f.name)
                save_file({k: (a * v + (1 - a) * t2[k]).contiguous() for k, v in t.items()},
                          args.out / "adapter" / f.name)
            else:
                save_file({k: (v * a if "lora_B" in k else v).contiguous() for k, v in t.items()},
                          args.out / "adapter" / f.name)
        else:
            shutil.copy(f, args.out / "adapter" / f.name)

    base = load_file((args.run2 / "readout.safetensors") if args.run2 is not None
                     else Path(resolve(args.asset)) / "readout.safetensors")["weight"].float()
    ft = load_file(args.run / "readout.safetensors")["weight"].float()
    save_file({"weight": ((1 - a) * base + a * ft).contiguous()}, args.out / "readout.safetensors")

    t_ft = float(json.loads((args.run / "calibration.json").read_text())["temperature"])
    t0 = float(json.loads((args.run2 / "calibration.json").read_text())["temperature"]) if args.run2 is not None else 1.0
    t = (1 - a) * t0 + a * t_ft
    (args.out / "calibration.json").write_text(json.dumps({"temperature": t, "wise_ft_alpha": a,
                                                           "source_run": str(args.run)}, indent=1))
    print(f"alpha {a}: adapter lora_B x {a}, readout {1 - a:.2f}*release + {a:.2f}*ft, T {t:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
