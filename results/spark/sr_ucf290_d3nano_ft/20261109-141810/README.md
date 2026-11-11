Local scoring stopped by Eric on 2026-11-09 (about 22:40 Taiwan time) to leave the GB10 to the floor2x_hn training. wise_a5/draft/scores.jsonl is partial; sr-score.py resumes it. colab/ holds the Colab setup.sh and score.sh (Drive folder s1-colab-wise-score).

2026-11-10: run-resume.sh finishes the draft, runs B and the evaluations (log: run-resume.log).

2026-11-11: run-resume.sh finished 2026-11-10T17:08Z (draft 16,812 windows resumed, 4361 s; B 18,390 windows, 13045 s, 0.709 s/window; peak 4.60 GB). Results: eval-vs-mixkl.log, eval-2x-vs-1x.log, score-shift-B.log, frame-auc.log, macro.log (macro AUC, results/spark/macro_auc_ucf290/20261110-133458/macro_auc.py).
