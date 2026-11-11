# Why 2x helps small events most, test 290 (2026-11-10)

Eric: "研究一下為何在small event上提升最多". CPU only, existing scores. `why_small.py` (docstring has the method),
`why_small.log`, `why_small.json`. Models: zero-shot d3-nano, mix + KL (floor_d3nano_c), fovfull2x, floor2x_hn;
1x = original 8 frames, 2x = all 8 frames bicubic 2x.

"adjusted" = difference of the mean 1x -> 2x score change between two groups at the same 1x score
(bins 0.2 wide), 95% interval from 1,000 draws of anomaly videos.

| Model | small - large raw | small - large adjusted | small - anomaly negative adjusted |
|---|---|---|---|
| zero-shot | +0.038 | +0.029 [-0.005, +0.066] | +0.044 [+0.023, +0.075] |
| mix + KL | +0.044 | +0.008 [-0.026, +0.039] | +0.044 [+0.015, +0.099] |
| fovfull2x | +0.042 | +0.022 [-0.020, +0.051] | +0.034 [+0.000, +0.094] |
| floor2x_hn | +0.024 | +0.022 [+0.002, +0.043] | +0.043 [+0.023, +0.068] |

small - medium adjusted: -0.018 to -0.001, all intervals include 0. Test set only; on the train scores Fable found
no event vs anomaly-negative difference within 1x bins (not rechecked here).
