# nano4 results, 2026-11-11 .. 11-14: image upsampling for small decision models in streaming VAD

Macro-AUC = mean per-video AUC. UCF-Crime: 290 test videos, fine-tuned d3-nano, three seeds (mean ± std).
ComplexVAD: 83 held-out test videos; MSAD: 221 test videos; both zero-shot / fine-tuned on UCF-Crime.
MSAD is a gated, research-only dataset: its per-window score files and training records are not in this repository;
only run configs, run.json and the aggregate numbers below.

| Input | tokens | ms (fast / ref) | UCF small-event AUC | UCF Macro | UCF AP | ComplexVAD zs / ft | MSAD zs / ft |
|---|---|---|---|---|---|---|---|
| whole frame 1x | 640 | 104 / 136 | 85.9 ± 0.5 | 76.7 ± 0.7 | 45.2 ± 2.1 | 78.5 / 80.8 | 78.6 / 83.2 |
| whole frame bicubic 2x | 2400 | 154 / 248 | 87.7 ± 0.3 | 80.5 ± 0.4 | 46.9 ± 1.6 | 82.1 / 82.4 | 80.5 / 86.7 |
| rectangle, separate image | 1015 | 122 / 169 | 87.5 ± 0.2 | 80.1 ± 0.3 | 46.4 ± 2.1 | 76.8 / 78.3 | - |
| rectangle, in place | 1015 | 112 / 168 | 86.7 ± 0.2 | 78.8 ± 0.4 | 46.4 ± 2.0 | 78.5 / 80.1 | 80.8 / 84.7 |
| one box per moving group | 854 | 114 / 159 | 86.7 ± 0.2 | 78.6 ± 0.5 | 46.2 ± 2.1 | 80.3 / 81.1 | 81.6 / 84.2 |
| moving patches | 783 | 109 / 147 | 86.8 ± 0.2 | 78.9 ± 0.3 | 46.0 ± 2.2 | 80.8 / 81.2 | 81.6 / 83.9 |
| most-changed patches, fixed budget | 1015 | 107 / 147 | 86.9 ± 0.3 | 79.6 ± 0.6 | 46.0 ± 1.9 | 81.9 / 82.2 | 79.2 / 81.2 |
| RLT (tau 0.1), before the tower | 562 | 108 / 142 | 86.0 ± 0.1 | 78.1 ± 0.1 | 46.3 ± 2.1 | 79.5 / 81.9 | 75.3 / 80.6 |
| CodecSight (0.25 px, our implementation), before the tower | 1060 | 136 / 188 | 87.6 ± 0.1 | 78.9 ± 0.6 | 46.1 ± 2.2 | 83.4 / 79.9 | 79.7 / 84.1 |
| FastV 50%, after the tower | 1200 | 163 / 220 | 87.2 ± 0.3 | 80.4 ± 0.4 | 46.3 ± 1.9 | 82.5 / 82.4 | 80.5 / 86.7 |
| FastV, 1015 tokens | 1015 | 152 / 209 | 87.1 ± 0.3 | 80.0 ± 0.5 | 46.1 ± 2.1 | 82.4 / 82.6 | 81.0 / 86.5 |
| VisionZip 70%, after the tower | 1680 | 149 / 222 | 87.7 ± 0.2 | 80.4 ± 0.4 | 47.2 ± 1.5 | 81.7 / 82.4 | 82.4 / 87.3 |
| VisionZip, 1015 tokens | 1015 | 149 / 203 | 87.5 ± 0.2 | 80.4 ± 0.3 | 47.5 ± 1.7 | 83.5 / 81.2 | 80.7 / 85.8 |
| FastV + our motion map | 1015 | - | 87.0 ± 0.1 | 79.8 ± 0.6 | 47.6 ± 1.3 | 80.7 / 81.3 | 81.8 / 84.3 |
| VisionZip + our motion map | 1015 | - | 87.1 ± 0.3 | 79.7 ± 0.6 | 47.5 ± 1.7 | 82.0 / 80.8 | 80.4 / 86.3 |
| first frame + moving cells (RLT / CodecSight + our motion map) | 513 | - | 84.7 ± 0.1 | 74.5 ± 0.4 | 44.3 ± 2.1 | 78.4 / 79.0 | 74.9 / 77.5 |

FastV, VisionZip and RLT: the selection rules of their official code ported to d3-nano (scripts/bl.py); CodecSight has
no released code and is re-implemented from the paper. Latencies of the published methods were measured in a second
job whose anchors read about 5 ms higher.

MSAD fine-tuned in domain (three seeds, 454 of 480 training videos; Macro-AUC): 180p 81.9 ± 1.5, bicubic 2x 84.4 ± 1.5,
real 360p 86.5 ± 1.2, rectangle in place 81.8 ± 1.8, one box per group 81.1 ± 2.4, most-changed patches 81.7 ± 1.7,
moving patches 83.6 ± 2.0.

Other small models, zero-shot, UCF-Crime 290 (scripts/smallvlm.py; visual tokens per frame at 1x -> 2x):
Qwen3-VL-4B 82 -> 302, Macro-AUC 71.8 -> 75.7. Gemma 3 4B (259 at both), Gemma 4 E4B (268), SmolVLM2-2.2B (1083),
InternVL3.5-2B (3331): runs in progress when this was written.

Scripts: scripts/bl.py (selections and named baselines), prune.py, smallvlm.py, tokevent.py, tokslope.py, split.py,
stream.py, gran.py, evalrun.py, cvad-metrics.py, cvad-prep.py, msad-prep.py, msad-train-prep.py. Job scripts as run on
the cluster: jobs/nano4-upx/.
