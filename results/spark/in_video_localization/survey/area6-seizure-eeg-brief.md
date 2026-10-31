# Brief: survey of patient-specific, causal seizure onset detection (EEG) methods we could reuse

## Area

Seizure onset detection from EEG or other physiological streams. The analogy: one patient = one scene. Signals differ a
lot between patients, events are short and rare, and detection must be online with few false alarms per hour.
Examples to start from: patient-specific vs cross-patient (patient-independent) detectors on CHB-MIT, TUSZ, Siena;
per-patient normalization or baseline subtraction; online adaptation to a new patient; transfer learning with a few
labelled seizures per patient.

## Questions

1. How do cross-patient detectors remove the patient baseline (per-recording normalization, subtracting a running
   baseline, contrastive or domain-adversarial training, patient-conditioned layers)? Which steps are causal?
2. How large is the gap between patient-specific and cross-patient detectors, and which methods close it?
   Give numbers (sensitivity, false alarms per hour, detection latency) and the dataset.
3. How does the field evaluate: event-based sensitivity, false alarms per 24 h, detection latency, any-overlap /
   epoch-based scoring (e.g., the SzCORE framework)? Which of these fit our alarm goal?
4. Which ideas can move to our setting: stored per-window features (1024-4096 dimensions), 2 s windows, video labels
   plus window labels, events of about 3 windows?
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
