# Survey summary: causal in-video event ranking (2026-10-29)

Six Sol surveys (`work brief --model sol`). Each `area*-brief.md` is the brief, `area*-sol.md` the report.
Shared context: `_context.md`. Numbers in the reports are cited by Sol and not yet checked one by one.
The first batch-3 run failed (ChatGPT 5-hour quota at 99 %, empty answer); logs in `failed/`. Rerun at 10:10 UTC.

| Area | Report | Top suggestion | Labels | On stored window features |
| --- | --- | --- | --- | --- |
| 1 Online action detection | `area1-online-action-detection-sol.md` | MiniROAD-style causal GRU head; GateHUB-style background gating; StartNet-style onset head | window | yes |
| 2 Prediction error / surprise | `area2-prediction-error-sol.md` | ridge / small predictor of the next window feature, score = prediction error | none (normal windows) | yes |
| 4 Change-point detection | `area4-change-point-sol.md` | one-sided CUSUM on p(yes); Scan B-statistic and online kernel CUSUM on features | none | yes |
| 9 Within-group ranking loss | `area9-within-group-ranking-sol.md` | same-video pairwise logistic loss + window BCE; PDAOM hard pairs; within-video partial AUC | window | yes |
| 6 Seizure detection (EEG) | `area6-seizure-eeg-sol.md` | novelty-gated causal baseline (past windows not flagged as events), raw + residual as inputs; scene-adversarial probe | window (+ video) | yes |
| 7 Speech online normalization | `area7-speech-online-normalization-sol.md` | causal scene mean shrunk toward a global normal prior; head conditioned on a past-scene summary (concat or FiLM) | window | yes |

## Common points

- No surveyed method reports per-video AUC or within-stream results; every gain must be measured here.
- Hard subtraction of a scene baseline is fragile; areas 6 and 7 both suggest feeding raw feature plus residual
  (or a scene summary) and letting the head learn the use.
- The start of a stream has no history: use a global normal prior (area 7) instead of the first window.
- Events are short (about 3 windows): baselines must not absorb the event (novelty gating, pause updates on high scores).
- Report per-video AUC, AUC_A, event sensitivity, false alarms per hour and onset latency together.
