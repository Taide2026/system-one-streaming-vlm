# run-1: probe 5 frozen vision encoders on the probe_ucf_short_v1_f8 windows (2026-10-28)

Proposal: encoder-probing (local `.specs/changes/2026-10-28-encoder-probing/proposal.md`). Config: `config.yaml`.
Smoke test and decision 9 check on 5 windows: `../smoke-1/`.

## Steps

`bash results/spark/encoder_probe_ucf_short_v1_f8/run-1/scripts/run.sh` (worktree root). Progress: `logs/run.log`.

1. `s1stream encoder-extract` for each encoder, one at a time: features in `$S1_FEATURES/encoder_probe_ucf_short_v1_f8/<folder>/run-1/`. Logs: `logs/extract-<encoder>.log`.
2. Decision 9 check on all `qwen35-vit` windows against vit-1 (`../smoke-1/scripts/check_equal.py run-1`). Log: `logs/check-equal.log`.
3. `s1stream probe-fit --device cuda` for each of the 7 feature folders: `fit-<folder>/`. Logs: `logs/fit-<folder>.log`.
4. `s1stream encoder-compare`: `encoder_compare.json`, `mil.png`, `holmes.png`, `knn.png`. Log: `logs/compare.log`.

## Status

Running.
