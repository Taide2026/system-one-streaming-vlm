# In-video localization on UCF-Crime (2026-10-29 to 2026-10-31)

Question: the probes and VLMs reach frame AUC 0.80-0.89, but can they find the event inside one anomaly video?
The goal is causal, per-window localization on a new camera (dev doc, research direction 1).

All analyses use stored scores and features; no model is loaded, except `06` (pixel motion from the videos, CPU).
They start from the encoder probing run `../encoder_probe_ucf_short_v1_f8/20261028-five-encoders/` (its `config.yaml`
defines the 117 short test videos: 49 anomaly, 68 normal; its `fit-*/` folders hold the probe fits).
Folder paths are in `scripts/paths.py`. Run every script from `scripts/`: `uv run python <script> ...`.
Features: `$S1_FEATURES/encoder_probe_ucf_short_v1_f8/<encoder>/run-1/` (encoders) and
`$S1_FEATURES/probe_ucf_short_v1_f8/A/run-1/` (Qwen3.5-9B hidden states, condition A).

Metrics (dev doc, section "計算方式總表"): "in-video" = AUC_A, event frames against the other frames of the anomaly
videos; "before vs normal" = AUC of the windows before the event against normal-video frames (0.5 = looks normal).
Selection: every layer, regularization and weight is chosen on validation; test is only reported. Exception noted in `07`.

These folders moved here from `encoder_probe_ucf_short_v1_f8/run-1/analysis-*` on 2026-10-31; old logs, commit
messages and survey briefs still show the old paths.

## Topics

| Folder | Question | Scripts | Answer |
|---|---|---|---|
| `01-frame-auc-split/` | Where does the 0.8 frame AUC come from? | `per_category_auc.py`, `per_video_auc.py`, `pyes_in_video_auc.py`, `llm_layer_in_video_auc.py`, `finetune_in_video_auc.py` | From separating anomaly videos from normal videos (event vs normal 0.91-0.99). In-video AUC 0.49-0.71 for every encoder, LLM state, zero-shot p(yes) and LoRA. Per-video median 0.686 (Qwen ViT MIL). |
| `02-generalization/` | Is the head too weak, or does it not transfer? | `train_in_video.py`, `mlp_in_video.py` | It does not transfer. Linear head in-video train 0.81-0.99, val/test 0.60-0.68; MLP fits train, val/test unchanged; `oracle_cv` (test labels, folds by video) 0.58-0.70. |
| `03-scene-removal/` | Does removing the scene from the features help? | `centered_fit.py`, `check_centered.py`, `residual_fit.py`, `smoothing_auc.py` | Whole-video mean subtraction 0.668 → 0.763 (V-JEPA 2) but uses the future; causal past-mean unstable (0.44-0.78); [x, residual] 0.62-0.67; smoothing lowers in-video AUC. |
| `04-loss-changes/` | Do loss or weighting changes teach the event? | `pairwise_fit.py`, `shortcut_fit.py` | No. Same-video ranking: no reliable gain. Ranking against before-event windows and JTT: test in-video drops. |
| `05-event-phase/` | Do anomaly videos look abnormal before the event? | `event_phase_auc.py` | Yes: before vs normal 0.82-0.95 (heads), 0.72-0.80 (zero-shot), 0.92-0.96 (LoRA). LoRA shrinks the during-minus-before gap to 0.02-0.05. All trained models learn the scene. |
| `06-motion-tempdiff/` | Does change or pixel motion avoid the scene? | `tempdiff_fit.py`, `motion_fit.py` | Training-free optical flow: before vs normal 0.48 (no shortcut), in-video 0.659, overall only 0.644. Heads on motion or on x_t − x_(t−1) learn the scene again or fail. |
| `07-fusion/` | Does head + flow keep both signals? A causal own-video anchor (NOVA)? | `fusion_anchor.py`, `fullflow_fusion.py` | On the 117 videos w = 0.2-0.3 gave +0.03-0.04 in-video, but w was picked on test (validation chose w = 0). Held-out 173 videos with VLM scores: +0.003 to +0.009 only. Anchor: no gain. |
| `08-bias-expert/` | Does a scene-only bias model (product of experts) absorb the shortcut? | `bias_expert_fit.py` | Encoder features: val and test agree, in-video +0.016 (V-JEPA 2) / +0.051 (Qwen ViT). Qwen3.5-9B hidden states: no gain. Not moved to LoRA. |

Shared helpers: `paths.py` (folders), `window_eval.py` (test report, BCE head with validation selection,
previous-window index). Each topic folder has a `logs/` folder with the printed output of its scripts.

## Survey

`survey/`: briefs (`*-brief.md`) and Sol reports (`*-sol.md`) of `telonium work brief --model sol`.
`summary.md` summarizes areas 1-9; `_context.md` is the shared context of those briefs.

| Report | Topic |
|---|---|
| `in-video-localization-sol.md` | First survey of in-video localization |
| `area1` … `area9` | Online action detection, prediction error, change point, seizure EEG, speech normalization, within-group ranking |
| `area10-event-vs-scene-shortcut-sol.md` | Methods that learn the event, not the scene (led to `04` before-event ranking and JTT) |
| `area11-what-will-work-sol.md` | Diagnosis from our evidence and a ranked plan (led to `07` fusion and the local-feature check) |
| `area12-deep-read-sol.md` | Four papers from other fields read closely: NOVA, Anomize, first-shot ASD, Contextual Matrix Profile (led to `07`) |

## Status

Open. Next step (dev doc): check whether pooling the window into one vector (patch mean, last token) loses local
events, with unpooled Qwen ViT patch features on a small set of windows. Needs a proposal (new extraction run).
Only Qwen-family features from now on (dev doc decision 11).
