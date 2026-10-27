"""Run `s1stream probe-fit` with a CUDA memory cap (resource rule) and report time and peak GPU memory.
Usage: fit_gpu.py <features run dir> <out dir> [cap GiB]"""
import sys
import time

import torch

from s1stream import cli

cap = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
torch.cuda.set_per_process_memory_fraction(cap / (torch.cuda.get_device_properties(0).total_memory / 2**30))
started = time.perf_counter()
code = cli.main(["probe-fit", sys.argv[1], "--out", sys.argv[2], "--device", "cuda"])
print(f"exit {code}, wall {time.perf_counter() - started:.1f} s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
sys.exit(code)
