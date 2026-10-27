# probe_ucf_short_v1_f8 / mlp-1

Config: `config.yaml` in this folder (the base config plus the settings of this run).

Derived experiment of `../vit-1` (2026-10-27): does a nonlinear read-out find more than the linear probe?

- Head: Linear(d → 256) → GELU → Dropout 0.1 → Linear(256 → 1). AdamW lr 1e-3, 1000 full-batch steps, weight decay in {1e-4, 1e-2, 1e-1}.
- Everything else as `probe-fit`: standardization, validation split, MIL top-k in {1, 3, frac8} on video labels, Holmes window head with pos_weight, choice on validation, refit on all train windows.
- GPU, memory cap 4 GiB; peak 0.49 GiB. 36–69 s per target.
- Note: the chosen weight decay is often 0.1, the edge of the grid.

## Steps

| Step | Command (worktree root, `uv run python`) | Log |
|---|---|---|
| Fit | `scripts/mlp_probe.py <name>=<features>:<kind>:<state> ...` | `logs/merger.log`, `logs/targets.log` |
| Eval | `scripts/eval_mlp.py` (same 1,000 video resamples as compare.json, paired with the linear probe) | `logs/eval.log` |

Outputs: `targets/<name>.npy` (test scores: mil, holmes), `targets/<name>.json` (choices), `eval_mlp.json`.

## Results (test frame AUC)

| Target | MIL linear | MIL MLP | MLP − linear | Holmes linear | Holmes MLP | MLP − linear |
|---|---|---|---|---|---|---|
| ViT state 12 mean | 0.795 | 0.790 | −0.005 [−0.015, +0.005] | 0.794 | 0.775 | −0.019 [−0.066, +0.021] |
| ViT state 27 mean | 0.825 | 0.819 | −0.006 [−0.015, +0.006] | 0.819 | 0.787 | −0.032 [−0.073, +0.010] |
| merger mean | 0.819 | 0.815 | −0.004 [−0.027, +0.015] | 0.744 | 0.790 | +0.046 [−0.011, +0.111] |
| LM state 0 visual (C) | 0.819 | 0.814 | −0.005 [−0.027, +0.013] | 0.738 | 0.795 | +0.056 [−0.003, +0.130] |
| LM state 18 visual (C) | 0.856 | 0.852 | −0.004 [−0.016, +0.007] | 0.784 | 0.820 | +0.036 [−0.003, +0.089] |
| LM state 23 last (C) | 0.851 | 0.841 | −0.009 [−0.022, +0.004] | 0.864 | 0.857 | −0.007 [−0.034, +0.014] |

Every MLP − linear interval contains 0. A stronger read-out on pooled features does not find more.
