# Brief: survey of training losses that rank items within a group

## Area

Loss functions and training schemes that optimize the ranking of items inside one group, not across groups. Any field:
learning to rank (query = group), recommendation, metric learning, multiple-instance learning, partial AUC / AUC
maximization, within-subject or within-patient ranking in medicine. Also VAD papers that use intra-video contrast or
ranking (e.g., the MIL ranking loss of Sultani et al., temporal contrastive or "intra-video" losses).

## Questions

1. Which losses directly optimize a within-group AUC or within-group ranking (pairwise, listwise, softmax over the
   group, AUC surrogate losses such as LibAUC)? Which need item-level labels, and which work with group labels only?
2. With window labels (HOLMES-VAU), how should pairs be built: event vs non-event windows of the same video only, or
   mixed with normal videos? Any evidence on the trade-off with cross-group (video-vs-video) separation?
3. With video labels only (MIL), which losses avoid the shortcut of ranking the whole video high (scene shortcut)?
4. Can these losses train a small head (linear, MLP, small temporal model) on our stored window features?
   Give the expected cost and pitfalls (few positives per video: about 3 event windows; class imbalance).
5. Any results that report a within-group metric (per-query NDCG, per-video AUC, per-subject AUC) before and after
   switching to a within-group loss?
## Context

We do streaming video anomaly detection on UCF-Crime. A model scores each 2 s window (8 frames, 2 s stride).
The deployment goal: one camera watches one scene for a long time and must raise an alarm when an event starts.
So the score may use only the current and earlier windows (causal).

What we found:
- Frame AUC over all 290 test videos is 0.82-0.89 (zero-shot VLM p(yes), LoRA fine-tunes, frozen-encoder probes).
- It mostly measures separating anomaly videos from normal videos (0.93-0.99).
- Localizing the event inside one anomaly video is only 0.57-0.74 (AUC_A, and the mean of per-video AUCs).
- Features of frozen vision encoders (Qwen3.5 ViT, SigLIP2, DINOv3, V-JEPA 2-L, InternViT) are dominated by the scene.
  Subtracting the mean feature of the whole video and refitting a linear probe raises in-video AUC 0.668 -> 0.763
  (V-JEPA 2), but this uses future windows.
- Causal versions (subtract the mean of the past 10 s, 20 s or all past windows) are unstable:
  kNN improves 0.50 -> 0.64, a window-label linear probe drops, a MIL probe jumps between 0.44 and 0.78.
- Temporal smoothing (Gaussian, moving mean, EMA) lowers in-video AUC.
- Labels: UCF-Crime train has video labels only. We also have window labels from the HOLMES-VAU annotations.
- Compute: one GB10 (unified memory). We have stored per-window features of 5 frozen encoders and LLM hidden states,
  so methods that work on stored per-window features are cheapest.

The abstract problem: within one stream of one scene, the features are dominated by the scene; we need a causal score
that ranks event windows above the other windows of the same stream.

## What to report for each method

| Field | Content |
| --- | --- |
| Method | name, paper title, venue, year, arXiv id |
| Idea | 1-2 sentences |
| Causal? | yes / no / can be made causal (how) |
| Labels needed | none / video / window or frame / other |
| Input | raw frames / per-frame or per-clip features / other |
| Fit for us | can it run on our stored 2 s window features? what changes are needed? |
| Code | GitHub URL, or "no code found" |
| Reported numbers | dataset, metric, value, table; plus any per-video, within-stream or latency result |

## Rules

- Reply with the content of the report file only (Markdown). No preamble.
- Cite every number with the paper and the table it comes from. Mark what you could not verify as "not verified".
- Do not invent papers, numbers or URLs. If you find nothing for a question, say so.
- Prefer 2023-2026 work, but include classic methods that are still the standard baseline.
- About 2 pages: a method table first, then short notes, then a ranked list of the 3 methods we should try first and why.
