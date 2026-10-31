# Brief: find and deep-read a few papers (any field) that could help causal in-video event localization

## Goal

Pick 4-6 papers from ANY field that could help our problem, read each one closely, and explain concretely what we
could take from it. Depth over breadth: do not list many papers. A paper from an unrelated field (medical signals,
speech, finance, industrial monitoring, robotics, remote sensing, NLP, recommendation) is welcome if its mechanism
transfers. Avoid repeating papers already covered in this folder (`summary.md`, `area*-sol.md`), unless you add a
deeper reading that the earlier report missed.

## Our problem in one paragraph

A camera watches one scene. Each 2 s window gets a score from frozen features (vision encoders, VLM hidden states)
or pixel motion. The score must be causal (current and past windows only). Every trained model learns "this footage
looks like an anomaly video" (scene, camera, quality) instead of "the event is happening now": windows BEFORE the
event already score far above normal videos (AUC 0.82-0.96), and in-video AUC (event vs other windows of the same
anomaly video) stays 0.53-0.71. Heads fit train videos (in-video 0.81-0.99) but drop to 0.60-0.68 on new videos;
an oracle fit on test labels with folds by video also stays 0.58-0.70. Loss tricks (same-video ranking, before-event
negatives, JTT) failed. Subtracting the whole-video mean helps (0.668 -> 0.763) but uses the future; causal
past-mean subtraction is unstable. Training-free pixel motion (mean optical-flow magnitude) has no scene shortcut
(before-event vs normal 0.48) and in-video 0.659, but cannot tell anomaly from normal videos (overall 0.644);
a head trained on motion features learns the scene again. About 390 train videos; labels: video-level plus window
labels (HOLMES-VAU) for train anomaly videos. Context file with more detail: `_context.md` in this folder.

Abstract form: within one stream, a strong, slowly varying nuisance (the scene / subject / sensor) dominates the
features and is correlated with the label across streams; the event is a short, local change. We need a causal score
that ranks event windows above the other windows of the same stream and holds on new streams.

## What to look for (examples, not limits)

- Methods that separate a stream-specific nuisance from a short event: per-subject / per-speaker / per-machine
  normalization, nuisance-invariant representation, subject-independent detection with few subjects.
- Evidence that a model learned a shortcut and a mechanism that removed it while improving within-group ranking.
- Combining a "which stream" signal with a "what changed now" signal (gating, two-stream, product of experts).
- Online / causal baselines that do not absorb short events.
- Any paper that reports a within-stream or within-subject metric and shows what raised it.

## For each paper (4-6 papers)

| Field | Content |
| --- | --- |
| Paper | title, venue, year, arXiv id or URL (must exist) |
| Field and problem | 1-2 sentences, and the mapping to ours (what is their "scene", "event", "stream") |
| Mechanism | what exactly they do, at the level we could reimplement (inputs, loss, update rule, key hyperparameters) |
| Evidence | the result that shows it works, with table number and values; say if it is within-stream / cross-subject |
| What we can take | the concrete change to our pipeline, which of our stored inputs it uses, and whether it stays causal |
| Risk | why it might not transfer to our case |

Then:
1. A short synthesis: which common idea appears across the papers, and which of our failed attempts it explains.
2. One recommended experiment per paper (cheapest version on our stored features or scores), with the metric and
   the result that would make us drop it.

## Rules

- Reply with the content of the report file only (Markdown). No preamble.
- Only real papers you can verify; give working URLs. Cite every number with paper and table; mark anything you
  could not verify as "not verified". Do not invent.
- Separate what the paper shows from your inference about our case.
- About 3-4 pages.
