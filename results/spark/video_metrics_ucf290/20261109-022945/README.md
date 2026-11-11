# video_metrics_ucf290 / 20261109-022945

Video-level metrics of the up-scale cascade on the 290 UCF-Crime test videos (Eric, 2026-11-09).
The score layouts are rebuilt with `score_gate_ucf390/20261108-155845/gates.py` and its config. The window AP column
reproduces that run (for example mix + KL: 47.3, 48.7, 56.7, anomaly head 48.5).
Metrics: `s1stream.metrics` and `s1stream.bootstrap.metric_summary` from `experiment/encoder-probing` (commit `196bdae`),
frame level (each frame takes its window score), paired video bootstrap, 1,000 draws, seed 20261022, against "original".

| File | Content |
|---|---|
| `config.yaml` | inputs and settings |
| `video_metrics.py` | the script |
| `run.log` | point values of every layout |
| `video_metrics.json` | point values, intervals and differences against original |

## Main rows (%), point values

| VLM | Layout | window AP | Macro AUC | within_auc | AUC_A | recall at 1 FA/h | frame AUC |
|---|---|---|---|---|---|---|---|
| zero-shot | original | 41.9 | 75.7 | 70.3 | 72.9 | 16.4 | 82.8 |
| zero-shot | 2x everywhere | 42.6 | 78.1 | 71.7 | 73.8 | 20.2 | 83.9 |
| zero-shot | oracle | 47.4 | 79.3 | 72.5 | 76.2 | 18.8 | 84.7 |
| zero-shot | anomaly head gate, replace | 42.8 | 76.9 | 70.4 | 73.4 | 18.6 | 83.1 |
| zero-shot | anomaly head gate, max | 43.2 | 77.5 | 70.4 | 73.6 | 18.1 | 83.3 |
| zero-shot | all windows, lambda 0.75 | 43.2 | 78.2 | 71.5 | 73.8 | 17.9 | 83.9 |
| mix + KL | original | 47.3 | 78.5 | 73.5 | 73.6 | 24.7 | 88.1 |
| mix + KL | 2x everywhere | 48.7 | 81.6 | 75.1 | 76.0 | 30.6 | 89.1 |
| mix + KL | oracle | 56.7 | 86.6 | 81.8 | 80.4 | 31.0 | 91.4 |
| mix + KL | anomaly head gate, replace | 48.5 | 80.5 | 73.9 | 74.6 | 29.4 | 88.4 |
| mix + KL | anomaly head gate, max | 48.5 | 80.7 | 73.8 | 74.6 | 27.8 | 88.4 |
| mix + KL | all windows, lambda 0.25 | 48.1 | 79.7 | 74.0 | 74.3 | 27.3 | 88.5 |

Macro AUC difference against original, 95% interval: zero-shot 2x everywhere +2.4 [+0.3, +4.5], anomaly gate max
+1.9 [+0.7, +3.3]; mix + KL 2x everywhere +3.0 [+1.5, +4.7], anomaly gate replace +2.0 [+0.6, +3.6], oracle +8.0
[+6.0, +10.3]. Recall at 1 FA/h intervals are wide (mix + KL 2x everywhere +5.9 [-9.1, +13.6]).
Band, rise and rise + motion rows are in `run.log`. The benefit heads are not included.
