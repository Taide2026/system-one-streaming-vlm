# system-one-streaming-vlm — branch `experiment/kv-across-windows`

A pilot on one question: **can a decision model that keeps the past across 2 s windows catch an event that is split
over two windows?** (Granularity case ③: RoadAccidents010, a 1.4 s event over the windows 6–8 s and 8–10 s;
Clef-Flash reads p(Yes) 0.26 and 0.20, both under 0.5, missed.)

This branch keeps only what the pilot needs: the `clef_flash` backbone (causal, so it can keep a KV cache), the
UCF-Crime and XD-Violence datasets, the frames input form, the `option_logits` head, the runner, the report and the
Docker / queue scripts. The other backbones, their configs and the earlier results live on `main`.

## Strategies (config section `context`)

The decision and its label stay on the newest 2 s window in every strategy, so every run is scored on the same frames.

| | `context` | What the model sees at window t | Cost per window |
|---|---|---|---|
| S0 today | (none) | the newest 2 s only | 8 frames |
| S1 recompute | `{strategy: recompute, past_s: L}` | the newest 2 s + the past L s, read again | (L/2 + 1) × 8 frames |
| S2 sliding KV | `{strategy: kv, past_s: L}` | as S1, but the past's KV / recurrent state is kept | 8 frames + the question |
| S3 sinks + sliding | `{strategy: kv_sinks, past_s: L, sinks_s: 2}` | S2 + the first 2 s never evicted (StreamingVLM) | 8 frames + the question |
| S4 compressed KV | `{strategy: kv_compress, past_s: L, budget_tokens: N}` | S2, the past windows pruned to N visual tokens (every other token, oldest first) | 8 frames + question + pruning |
| S5 text memory | `{strategy: text, history: n}` | the newest 2 s + the last n scores written into the state | 8 frames + a few tokens |

`src/s1stream/context.py` defines the section; `src/s1stream/runner.py` applies it; the KV strategies need a
`StreamingBackbone` (`src/s1stream/slots.py`), which `clef_flash` implements with `src/s1stream/kvcache.py`.

### How S2 works in `clef_flash`

The release lays a record out as `prefix | frames | "\n" | state | schema | suffix`. Streaming keeps one
`DynamicCache` per video: the prefix is prefilled once; each window prefills **only its 8 new frames** on top of the
cached frames; the question (`"\n" + state + schema + suffix`) is appended, the joint schema head reads the hidden
states of every cached token plus the question's, and the cache is restored to the snapshot taken before the question.
Positions are the 3D M-RoPE positions of the whole history, so while nothing has been evicted the result is the
release's forward on the same frames (checked against `encode_record` on the first window of every video). Frames
older than `past_s` are removed from the full-attention layers' keys and values; the Gated DeltaNet layers (3 in 4 of
Qwen3.5) hold a fixed-size recurrent state that cannot forget, so their trace of evicted frames stays.

## The pilot (about 10 videos)

`configs/pilot_ucf_v1_*.yaml`: the same 10 UCF-Crime test videos, V1 prompt, 2 s windows of 8 frames; only `context`
differs. 5 split events shorter than 4 s that Clef-Flash missed in both windows (RoadAccidents010 / 004 / 011 / 122,
Shoplifting037) and the 5 shortest normal test videos.

```bash
scripts/run.sh setup                       # uv sync, CPU tests
scripts/run.sh assets                      # clef-flash; ucf-crime from the ucf-clips download
scripts/run.sh smoke configs/pilot_ucf_v1_s2_kv_l2.yaml   # 5 windows
scripts/run.sh pilot                       # every pilot config, then the two checks below
```

What is measured (`scripts/compare-runs.py`, `scripts/pilot-report.py`):

- **Correctness first:** S2 with L = 2 must give S1's probabilities on the windows 0–1 of each video (nothing evicted
  yet). bf16 numerics put the difference around 1e-2 at most; more is a bug.
- **Effect on case ③:** p of the event windows per strategy, how many of the 5 split events are caught at 0.5 (and
  whether by a later window), FPR at 0.5 on the normal videos' windows; frame AUC / AP as before (`report.json`).
- **Cost:** latency P95 and RTF P95 (≤ 1), peak GPU memory, per strategy and L.

Go on to all 290 / 800 videos only if S2 = S1 holds and more split events are caught at RTF ≤ 1.

## Everything else

Setup, assets, the UCF-Crime layout, the run folder, the metrics and the Docker / queue scripts are as on `main`
(see its README). `s1stream run` writes `frames_read` into `scores.jsonl` for the recompute strategy, the number of
frames the model actually saw. Tests: `uv run pytest -q` (CPU, no model).
