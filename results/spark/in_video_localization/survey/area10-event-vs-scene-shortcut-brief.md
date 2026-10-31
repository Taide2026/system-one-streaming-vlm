# Brief: survey of training methods that make a model learn the event itself, not "this footage looks like an anomaly video"

## Area

Shortcut learning and scene bias, applied to temporal event detection in video. Relevant fields:
- Scene-debiased action recognition: background erasing, background or actor mixing, scene-adversarial losses,
  static-frame augmentation (examples to start from: SDN "Why Can't I Dance in the Mall?", BE "background erasing",
  FAME, ActorCutMix, StillMix). Check that each exists before you cite it.
- Spurious-correlation robustness: group DRO, JTT, counterfactual or copy-paste augmentation, invariant risk
  minimization, gradient reversal / domain-adversarial training with the video or scene as the domain.
- Video anomaly detection methods that model the scene or the camera explicitly (scene-aware, scene-conditioned,
  per-camera normalization), and weakly supervised VAD methods that report in-video localization (AUC_A, per-video AUC).
- Temporal / change-based objectives: contrastive losses between windows of the same video, temporal difference or
  motion features, hard negatives from the same video.

## Questions

1. Which methods make a detector depend on the event (action, motion, change) and not on the scene or footage type?
   For each: is it a data method (augmentation, sampling), a loss, or an architecture change?
2. Which of them were shown to raise a within-video or cross-scene metric (not only the pooled metric)?
   Give numbers with dataset and table.
3. Which can be applied to (a) a LoRA fine-tune of a VLM that answers Yes/No per 2 s window, and
   (b) a linear or small head on stored per-window features? What does each need (labels, extra data, masks)?
4. How does each field test that the shortcut is gone (e.g. scene-swapped test sets, mimetics, background-only
   inputs)? Which test could we run on UCF-Crime with our data?

## Context

We do streaming video anomaly detection on UCF-Crime. A model scores each 2 s window (8 frames, 2 s stride).
The deployment goal: one camera watches one scene for a long time and must raise an alarm when an event starts.
So the score may use only the current and earlier windows (causal).

What we found:
- Frame AUC over all 290 test videos is 0.82-0.89 (zero-shot VLM p(yes), LoRA fine-tunes, frozen-encoder probes).
- It mostly measures separating anomaly videos from normal videos (0.93-0.99).
- Localizing the event inside one anomaly video is only 0.57-0.74 (AUC_A, and the mean of per-video AUCs).
- Event-phase test: windows of anomaly videos BEFORE the event already score far above normal videos.
  Zero-shot VLMs: before 0.72-0.78, during 0.88-0.93, after 0.77-0.85 (AUC against normal-video frames).
  8 LoRA fine-tunes: before 0.92-0.96, during 0.96-0.99, after 0.92-0.97. The during-minus-before gap shrinks from
  0.13-0.16 (zero-shot) to 0.02-0.05 (fine-tuned), and AUC_A stays at 0.72-0.73.
  So the fine-tune learned "this footage looks like an anomaly video" (camera type, place, quality), not the event.
- The LoRA labels are window-level (HIVAU-70k / HOLMES window coverage, binarized at 0.5); 10k abnormal and 10k normal
  windows; normal windows come from normal videos and from non-event stretches of anomaly videos. One epoch,
  LoRA r64, frozen vision encoder, Yes/No prompt. So the shortcut is not caused by video-level labels.
- Frozen-encoder probes (Qwen3.5 ViT, SigLIP2, DINOv3, V-JEPA 2-L, InternViT): features are dominated by the scene.
  Subtracting the whole-video mean feature raises in-video AUC 0.668 -> 0.763 (V-JEPA 2), but uses future windows.
  Causal past-mean subtraction is unstable. Smoothing lowers in-video AUC. An MLP head fits train (in-video AUC ~1.0)
  but val/test stay at 0.58-0.68. A head fit on test ground truth with 5-fold by video (oracle_cv) also stays at
  0.58-0.70. A same-video pairwise ranking loss did not help reliably.
- Labels: UCF-Crime train has video labels only; we also have HOLMES-VAU window labels for train.
- Compute: one GB10 (unified memory). Stored per-window features of 5 frozen encoders and LLM hidden states exist,
  so methods that work on stored features are cheapest; LoRA runs are possible on a second machine.

The abstract problem: the scene / footage type predicts the video label; we need a training signal that forces the
score to rise only when the event happens, and that holds on new cameras.

## What to report for each method

| Field | Content |
| --- | --- |
| Method | name, paper title, venue, year, arXiv id |
| Idea | 1-2 sentences |
| Type | data / loss / architecture |
| Causal? | yes / no / can be made causal (how) |
| Labels needed | none / video / window or frame / other (masks, boxes) |
| Fit for us | LoRA VLM, stored-feature head, or both; what changes are needed |
| Code | GitHub URL, or "no code found" |
| Reported numbers | dataset, metric, value, table; mark within-video or cross-scene results |

## Rules

- Reply with the content of the report file only (Markdown). No preamble.
- Cite every number with the paper and the table it comes from. Mark what you could not verify as "not verified".
- Do not invent papers, numbers or URLs. If you find nothing for a question, say so.
- Prefer 2023-2026 work, but include classic methods that are still the standard baseline.
- About 2-3 pages: a method table first, then short notes per question, then a ranked list of the 3 methods we should
  try first, with the cheapest experiment for each and the metric that would show success (AUC_A, per-video AUC,
  event-phase during-minus-before gap).
