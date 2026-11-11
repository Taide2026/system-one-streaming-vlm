# benefit_gate_ucf390 / 20261108-145639

Zoom-benefit gate heads on V-JEPA 2 features (layer 18 patch_mean). CPU only. Run on the GB10, 2026-11-08.

Question: does a head trained to predict "the 2x up-scale helps this window" gate the VLM better than the
anomaly/normal head?

## Files

| File | Content |
|---|---|
| `config.yaml` | Config of this run (train scores, test features, head, weights, folds, upscale rate) |
| `benefit.py` | Labels, weights, 5-fold CV, final heads, test evaluation |
| `run.log` | Full log: CV, test AUC/AP per tier with 95% bootstrap CI of the AP change, zoom rates |
| `benefit.json` | All numbers of `run.log` |
| `check-mixkl/` | Check run with the mix+KL label model only (same numbers as the full run) and two diagnostics |
| `check-mixkl/label_auc.py`, `label_auc.log` | Cross-fitted AUC of each head for b > 0, b-weighted AUC, Holmes AUC |
| `check-mixkl/mlp_check.py`, `mlp_check.log` | Linear vs MLP (hidden 256), in-sample and held-out AUC |

Train scores: `results/spark/train_zoom_scores_ucf390/20261108-131218/g4/<model>/<1x|2x>/scores.jsonl`
(RTX PRO 6000 on Colab, 390 training videos, 11,088 windows).

## Command

    timeout 3600 uv run --frozen python results/spark/benefit_gate_ucf390/20261108-145639/benefit.py \
      --config results/spark/benefit_gate_ucf390/20261108-145639/config.yaml > results/spark/benefit_gate_ucf390/20261108-145639/run.log 2>&1

## Results (AP all, test 290 videos, 18,390 windows)

| Head | gate zero-shot | gate mix+KL |
|---|---|---|
| original | 41.9 | 47.3 |
| all 2x | 42.6 | 48.7 |
| oracle (events only) | 47.4 | 56.7 |
| zero-shot label, best (logistic / crossing) | 42.2 | 47.9 |
| zero-shot label, main = ridge | 41.8 | 47.5 |
| mix+KL label, best (logistic) | 42.8 | 48.3 |
| mix+KL label, main = abs_b | 42.3 | 47.8 |

Main weight = highest 5-fold CV gate AP on the training videos (rule fixed before the run).
Diagnostics (mix+KL label): held-out AUC for b > 0 is 0.54-0.59 (linear) and 0.606 (MLP); in-sample 0.82 / 0.96.
