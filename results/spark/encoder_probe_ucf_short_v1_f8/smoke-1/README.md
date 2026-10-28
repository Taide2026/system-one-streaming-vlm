# smoke-1: resource smoke test and decision 9 check (2026-10-28)

Config: `config.yaml` (the encoder config plus a `run:` section). Features: `$S1_FEATURES/encoder_probe_ucf_short_v1_f8/<encoder>/smoke-1/`.

## Steps

1. `uv run s1stream encoder-extract configs/encoder_probe_ucf_short_v1_f8.yaml --encoder <name> --run smoke-1 --limit-windows 5`, one encoder at a time, `timeout 900`. Logs: `logs/extract-<encoder>.log`.
2. `uv run python results/spark/encoder_probe_ucf_short_v1_f8/smoke-1/scripts/check_equal.py smoke-1`: decision 9 check against vit-1. Log: `logs/check-equal.log`.
3. `uv run python results/spark/encoder_probe_ucf_short_v1_f8/smoke-1/scripts/attention_backend.py`: why step 2 failed. Log: `logs/attention-backend.log`.

## Resources (5 windows per set, GB10, bfloat16)

| Encoder | Peak GPU (GiB) | Max RSS (GB) | Load (s) | Test s / window | States | Dims | Head folder |
|---|---|---|---|---|---|---|---|
| qwen35-vit | 1.16 | 2.7 | 11.0 | 0.245 | 28 | 1152 | — |
| internvit-300m | 1.17 | 1.8 | 5.9 | 0.275 | 25 | 1024 | 1 state, 1536 dims |
| siglip2-so400m | 1.30 | 3.3 | 2.2 | 0.297 | 28 | 1152 | 1 state, 1152 dims |
| dinov3-vitl | 0.70 | 2.6 | 2.2 | 0.192 | 25 | 1024 | — |
| vjepa2-vitl | 0.71 | 2.7 | 2.1 | 0.210 | 25 | 1024 | — |

All peaks are far below the expected 10 GiB and the 24 GiB cap. 12,772 windows x about 0.25 s is about 55 minutes per encoder.
Before the GPU runs, every model tensor of the 5 loaded encoders was compared with its checkpoint tensor on the CPU: 0 differences.

## Decision 9 check: failed as written

`qwen35-vit` through `encoder-extract` against vit-1 (first 5 windows of each set): max abs difference / max abs value is
1.5e-4 to 9.7e-3 (threshold 1e-3). Shapes and the window plan are equal.

Cause (step 3, 5 test windows):

| Comparison | Min cosine patch_mean | Min cosine patch_max | Per-state ratio |
|---|---|---|---|
| ours bfloat16 vs vit-1 (bfloat16) | 0.999994 | 0.999952 | 0.4 to 1.2 % |
| ours bfloat16 vs ours float32 | 0.999979 | 0.999934 | 0.3 to 1.5 % |
| ours float32 vs vit-1 | 0.999965 | 0.999956 | — |

- The attention backend is not the cause: the default backend and the d3 runtime setting (no cuDNN) give equal features (difference 0).
- The difference to vit-1 equals the bfloat16 rounding of the same code (bfloat16 vs float32). bfloat16 has a relative step of 2^-8 = 3.9e-3, so a threshold of 1e-3 cannot pass.
- The threshold in decision 9 is wrong, not the code path. Status: waiting for Eric's decision on a new criterion.
