# Why frame AP < window AP (2026-11-11)

frame_vs_window.py <manifest> <scores.jsonl>: positive rates, boundary frames (label 0 frames inside a label-1 window), frame AP without them.

| Scores | window AP | frame AP | frame AP without boundary negatives |
|---|---|---|---|
| mix + KL 1x | 47.3 | 45.2 | 46.6 |
| floor2x 2x | 49.8 | 47.6 | 49.4 |

Positive rate: window 8.4 % (label 1 if any overlap), frame 7.6 %. Boundary frames: 9,065 (0.89 % of negative frames), mean score 0.564 (mix 1x) / 0.586 (floor2x 2x) vs 0.200 / 0.208 for other negatives.
