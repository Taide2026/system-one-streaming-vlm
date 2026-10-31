# Brief: survey of in-video (anomaly-only) localization metrics in video anomaly detection

Reply with the content of the report file only (Markdown). No preamble.

## Context

We do weakly supervised / tuning-free video anomaly detection (VAD) on UCF-Crime with 2 s windows.
The standard metric is frame-level AUC over all 290 UCF-Crime test videos (140 anomaly + 150 normal).
We found this metric mixes two things:
- separating anomaly videos from normal videos (event frames vs frames of normal videos): our models reach 0.97-0.99;
- localizing the event inside an anomaly video (event frames vs the other frames of the anomaly videos): only 0.70-0.74,
  for zero-shot VLMs and for LoRA fine-tuned VLMs alike (frame AUC 0.82 -> 0.89 after fine-tuning, in-video unchanged).

We want to know whether the field measures the second quantity, and how good the state of the art is on it.

## Questions

1. Which papers report an anomaly-only metric on UCF-Crime (and XD-Violence, ShanghaiTech if relevant)?
   Names used include AUC_A, AUC_abn, "abnormal AUC", "anomaly-video AUC", AP_A, "AUC on abnormal videos only".
   For each: paper, venue, year, exact metric definition, the reported number(s), and the overall AUC of the same model.
2. Which papers say the standard frame AUC is biased or inflated by normal videos (video-level separation),
   or propose other localization metrics (TIoU, mAP@tIoU on UCF-Crime / UCA, event-level metrics, per-video AUC,
   detection-delay or false-alarm-per-hour metrics for streaming)? Give their findings and numbers.
3. What is the best known anomaly-only AUC on UCF-Crime (weakly supervised, and separately tuning-free / zero-shot
   / VLM-based)? Give a small ranked table with the source of each number.
4. Is there any paper that explains the gap (e.g., scene bias, video-level label shortcut, MIL top-k behaviour) and
   proposes a fix (e.g., per-video normalization, background subtraction, scene-debiasing, temporal contrast)?

## Rules

- Cite every number with the paper (title, arXiv id or venue) and the table it comes from.
- Mark anything you could not verify from the paper text as "not verified".
- Do not invent numbers. If no paper reports a metric, say so.
- Keep it to about 1-2 pages: tables first, then short notes.
