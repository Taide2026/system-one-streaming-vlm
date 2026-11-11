# floor2x_hn vs mix + KL: Macro AUC and score by distance to the event (2026-11-10)

Question (Eric): is the weak floor2x_hn result a data problem or a loss problem? Two CPU checks on existing scores.
Scores: `sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c/{draft,B}` and `sr_ucf290_d3nano_ft/20261110-023316/{draft,B}`.
Run: `diagnose.py` (command in this README's git history; arguments mix1x, hn1x, mix2x, hn2x). Output: `diagnose.log`.

- Macro AUC is window level here (139 anomaly videos with both labels); the deck's 78.5 -> 81.6 for mix + KL uses another convention.
- Distance = windows to the nearest event window inside the same anomaly video.
