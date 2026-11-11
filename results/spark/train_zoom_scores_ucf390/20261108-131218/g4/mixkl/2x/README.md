mix+KL 2x scores of the 390 train videos, Colab G4, 2026-11-08.
scores-raw.jsonl has 21,720 lines: a restart left the first sr-score.py process running, so two processes scored
the same file (456 windows once, 10,632 twice). The duplicate scores are exactly equal (max difference 0).
scores.jsonl keeps the first line of each (video, window): 11,088 windows. Two processes shared the GPU, so the
per-window seconds of this run (about 0.27 s) are not the G4 speed of one process.
