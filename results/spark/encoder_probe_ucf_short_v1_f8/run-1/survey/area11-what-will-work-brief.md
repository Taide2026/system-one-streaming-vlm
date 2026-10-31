# Brief: which family of methods is likely to fix causal in-video event localization, given our evidence?

## Goal

We tried many single fixes and most failed. We need a diagnosis-driven plan, not another list of methods.
Use the evidence below to decide which method families can address the cause and which cannot. Rank them.
Do not repeat the earlier surveys: read `summary.md` and the `area*-sol.md` reports in this folder first
(online action detection, prediction error, change point, within-group ranking, seizure EEG, speech normalization,
event-vs-scene shortcut). Refer to them by name, and add only what they miss.

## Setting

Streaming video anomaly detection on UCF-Crime. A score per 2 s window (8 frames, 2 s stride). Deployment: one
camera watches one scene; the score may use only the current and earlier windows (causal); alarms must start soon
after the event begins with few false alarms per hour. Main goal: localize the event inside a video from a NEW camera.
Data: UCF-Crime train (video labels; HOLMES-VAU window labels for train anomaly videos), test with event frames.
Short test set: 117 videos (49 anomaly, 68 normal). Full test: 290 videos.
Compute: one GB10 (unified memory). Stored per-window features: 5 frozen vision encoders (Qwen3.5 ViT, InternViT,
SigLIP2, DINOv3, V-JEPA 2-L; patch mean), Qwen3.5-9B / d3-flash LLM hidden states, 61 pixel-motion values per window.
LoRA fine-tunes of 9B VLMs are possible on a second machine.

## Evidence (all numbers are ours, short test set unless marked)

Metrics: "in-video" = AUC_A (event frames vs other frames of the anomaly videos); "before vs normal" = AUC of the
windows before the event against normal-video frames (0.5 = looks like normal footage).

1. Frame AUC 0.80-0.89 everywhere, but it is mostly "anomaly video vs normal video" (event vs normal 0.91-0.99).
   In-video AUC is 0.53-0.71 for every encoder, LLM state, probe and zero-shot p(yes).
2. Event phase: windows BEFORE the event already score far above normal videos (encoder heads 0.82-0.95,
   zero-shot VLM 0.72-0.80). LoRA fine-tunes (window labels, 10k/10k windows) push "before" to 0.92-0.96 and the
   during-minus-before gap shrinks from 0.13-0.16 to 0.02-0.05 (full test). In-video stays 0.72-0.73 (full test).
   So all trained models learn "this footage looks like an anomaly video" (scene / camera / quality).
3. Generalization: linear head in-video AUC train 0.81-0.99, val 0.60-0.67, test 0.60-0.68. MLP fits train (~1.0),
   val/test unchanged. oracle_cv (fit on test GT, 5 folds by video) 0.58-0.70. So not label noise, not head capacity.
4. Scene removal on features: subtract whole-video mean 0.668 -> 0.763 (non-causal); causal past-mean subtraction
   unstable (0.44-0.78); [x, residual] 0.62-0.67; smoothing lowers in-video AUC.
5. Loss changes on stored features: same-video pairwise ranking (all non-event negatives) no reliable gain;
   ranking against before-event windows only lowers "before vs normal" (0.82 -> 0.66) but also "during", in-video
   0.668 -> 0.638-0.656; JTT upweighting hurts (in-video 0.51-0.59, overall AUC drops below 0.5 at weight 20).
6. Temporal difference d_t = x_t - x_(t-1): head on d_t in-video 0.56-0.59; head on [x, d] 0.62; training-free
   |d_t| 0.64 with the largest during-minus-before gap (0.19-0.22) and before vs normal ~0.5.
7. Pixel motion (Farneback flow on the 8 frames, no model): training-free mean flow magnitude in-video 0.659,
   before vs normal 0.48 (no scene shortcut), overall only 0.644. A head on 61 motion values: in-video 0.634 and
   before vs normal rises to 0.59 (the head learns scene again). [x, motion] concat: +0.01.
8. Literature: no surveyed method reports per-video AUC or within-stream results; DSS+ASR (WACV 2025) is the only
   one with a UCF-Crime AUC_A gain (64.0 -> 69.4); GS-MoE AUC_A 83.86 is non-causal.

## Questions

1. Diagnosis: from this evidence, what is the most likely cause of the low in-video AUC? Consider at least:
   (a) features dominated by scene so event signal is a small residual; (b) too few training scenes (about 390
   videos) to learn a scene-invariant event detector; (c) supervision rewards the video-level shortcut;
   (d) 2 s windows / 8 frames / patch-mean pooling lose the event (small, short, local); (e) UCF-Crime annotation
   or event definition limits (event boundaries, events that are mostly static). Say which evidence supports or
   rules out each. Say what cheap check would separate the remaining causes.
2. For each cause that remains, which method families attack it directly? Examples to judge (add others):
   spatially local features (patch-level, object or person tracks, pose) instead of patch mean; motion-centric or
   two-stream models; score fusion of a scene-level score and a motion / change score; causal sequence models over
   windows; prediction-error / novelty against the camera's own past; test-time or per-camera adaptation;
   more scene diversity (other datasets, synthetic event transplant); different supervision (frame-level, MIL with
   in-video ranking); VLM-side changes (prompt that asks about change, multi-window context, LoRA with in-video
   negatives).
3. Which families have published evidence on within-video or cross-scene localization (any domain: VAD, online
   action detection, temporal action localization, surveillance event detection)? Give paper, venue, year, table,
   number. Mark "not verified" where you cannot check.
4. Rank the top 3 directions for us. For each: why it addresses the cause, the cheapest decisive experiment with our
   stored data, the expected result if the hypothesis is right, and the result that would make us drop it.
   Prefer directions that keep the score causal and real-time.

## Rules

- Reply with the content of the report file only (Markdown). No preamble.
- Cite every number with the paper and table. Do not invent papers, numbers or URLs. Mark unverifiable items.
- Separate what our evidence shows from your inference.
- About 3 pages: diagnosis table first, then the method-family table, then the ranked plan.
