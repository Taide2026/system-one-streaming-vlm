# floor2x_hn_d3nano_c

Mix + KL (`floor_d3nano_c` recipe) retrained with hard negatives and 2x inputs (Eric, 2026-11-09).
Question: does the retrained model keep the 2x scores of anomaly-video negatives low, so the large-tier AP does not
drop at 2x (`floor_d3nano_c`: 29.0 -> 25.9; `fovfull2x_d3nano_c`: 22.5 -> 16.4)?
Config: `configs/floor2x_hn_d3nano_c.yaml`. Code: `scripts/train-floor.py` (hard_negative_share, upscale share,
frames cache, `--extract-frames`, S1_MEM_FRACTION, peak print); tests `tests/test_train_floor_upscale.py`.

| Group (v1) | train | train at 2x | val | val at 2x |
|---|---|---|---|---|
| events (y > 0) | 5,000 | 1,500 | 1,000 | 300 |
| anomaly-video negatives | 2,500 | 750 | 215 | 64 |
| normal-video negatives | 2,500 | 750 | 785 | 236 |

Train: 1,126 videos. Val is drawn at random (no hard negatives), so 21.5 % of its negatives come from anomaly videos.
c14 (5,000) uses the same windows at 1x; replay (5,000) is text only.

| Folder / file | Content |
|---|---|
| `extract-20261109-075346/` | config, `records.jsonl`, `val.jsonl` (GB10, `--extract-frames`, 315 s) |
| `extract-20261109-075346.log` | extraction log |
| `colab/setup.sh`, `colab/train.sh` | Colab G4 steps (Drive folder `s1-colab-floor2x_hn_d3nano_c`) |

Frame cache: `$S1_FEATURES/frames/floor2x_hn_train_f8/20261109-075346/` (12,000 .npy, 21 GB);
packed in `...-upload/` (10 tar.zst, 8.8 GB, uploaded to Drive 2026-11-09, sizes checked).

GB10 smoke test (`results/smoke/floor2x_hn_d3nano_c/20261109-075916`, `--limit-steps 5`, S1_MEM_FRACTION 0.5):
exit 0, 5 steps + temperature on 2,000 val windows in 31 min (15:59-16:30 Taiwan time), peak GPU 14.0 GB, T = 0.61.
The split between training steps and validation is not logged (train.log writes every 10 steps).

## Attempts (2026-11-09)

- Colab: G4 and A100 returned 503 Service Unavailable (no free machine); H100 was rejected (no entitlement for this account).
  No session was created, no compute units used.
- GB10 full run `20261109-090353/` (S1_MEM_FRACTION 0.5, timeout 36h): started 17:04 Taiwan time, stopped by Eric's
  request after about 5 minutes, before step 10. Not a result; the folder holds only config and records.
