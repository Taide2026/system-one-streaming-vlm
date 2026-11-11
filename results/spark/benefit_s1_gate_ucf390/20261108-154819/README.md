# benefit_s1_gate_ucf390 / 20261108-154819

Zoom-benefit gate heads with the VLM 1x score as an extra input. CPU only. GB10, 2026-11-08 (Eric approved).

Question: does the gate pick better windows when it also sees the VLM's own 1x score?
Same as `results/spark/benefit_gate_ucf390/20261108-145639/` except the head input = V-JEPA 2 features (1024) +
the 1x score of the label model (1 column). A head trained on model M's labels reads M's 1x score, so it gates
only M (10 results, not 20).

| File | Content |
|---|---|
| `config.yaml` | Config (copy of the benefit_gate config, new header) |
| `benefit.py` | Unchanged copy of the benefit_gate script (labels, heads, CV); imported by `benefit_s1.py` |
| `benefit_s1.py` | Adds the 1x score column; CV, online cutoff, test evaluation |
| `run.log`, `benefit_s1.json` | Results |

    timeout 3600 uv run --frozen python benefit_s1.py --config config.yaml > run.log 2>&1   # from this folder

## Results (AP all, test 290 videos)

| Head | gate zero-shot | gate mix+KL |
|---|---|---|
| original / all 2x / oracle | 41.9 / 42.6 / 47.4 | 47.3 / 48.7 / 56.7 |
| main (zero-shot: ridge; mix+KL: crossing) | 41.8 | 48.1 |
| best (zero-shot: logistic; mix+KL: logistic) | 42.4 | 48.5 |

The train-side cutoff up-scales 8.9-12.6% of the test windows for mix+KL and 17.0-27.2% for zero-shot (target 20%).
