"""Temporal smoothing of window scores inside each video, then all / vs_normal / in_video / per-video AUC.

HeadHunt-VAD (arXiv 2512.17601) smooths segment scores with a 1D Gaussian (sigma 1.5 segments) and reports
+4.59 frame AUC. This checks how much smoothing changes the in-video AUC of our scores. No model is run.
Smoothers (per video, windows in time order, 2 s stride):
  none
  gauss<s>  scipy.ndimage.gaussian_filter1d, sigma s windows, mode "nearest" (uses later windows: not causal)
  mean<k>   mean of the current and k - 1 earlier windows (causal)
  ema<a>    s_t = a * x_t + (1 - a) * s_(t-1), s_0 = x_0 (causal)
Usage: uv run python <this file>
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

import s1stream.plugins  # noqa: F401  (registers the datasets)
from s1stream.encoder.compare import select_state
from s1stream.encoder.extract import load_encoder_config
from s1stream.probe.compare import _weighted_curve, _window_counts
from s1stream.registry import DATASETS

RUN = Path(__file__).resolve().parents[1]
MAIN = Path("/home/cctsai/Eric/VLM/system-one-streaming-vlm/results/spark")
SMOOTHERS = ["none", "gauss1", "gauss1.5", "gauss2", "mean2", "mean3", "mean5", "ema0.5", "ema0.3"]


def smooth(x: np.ndarray, name: str) -> np.ndarray:
    if name == "none":
        return x
    if name.startswith("gauss"):
        return gaussian_filter1d(x, float(name[5:]), mode="nearest")
    if name.startswith("mean"):
        k = int(name[4:])
        c = np.concatenate([[0.0], np.cumsum(x)])
        i = np.arange(len(x))
        lo = np.maximum(i - k + 1, 0)
        return (c[i + 1] - c[lo]) / (i + 1 - lo)
    if name.startswith("ema"):
        a = float(name[3:])
        out = np.empty_like(x)
        out[0] = x[0]
        for t in range(1, len(x)):
            out[t] = a * x[t] + (1 - a) * out[t - 1]
        return out
    raise ValueError(f"unknown smoother: {name}")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def probe_scores(fit_dir: Path, method: str, kind: str) -> tuple[list[dict], np.ndarray]:
    """Test meta rows and the test window scores of one probe at the state chosen on validation."""
    fit = json.loads((fit_dir / "fit.json").read_text())
    state = select_state(fit, method, kind)
    scores = np.load(fit_dir / "test_scores.npy", mmap_mode="r")
    values = scores[fit["methods"].index(method), fit["kinds"].index(kind), state]
    return read_jsonl(fit_dir / "test_meta.jsonl"), np.asarray(values, dtype=np.float64)


def sources():
    """(name, meta rows, scores) for each score source on the 117 test videos."""
    pyes_file = sorted((MAIN / "d3_flash_ucf_short_f8").glob("*/scores.jsonl"))[-1]
    pyes_rows = read_jsonl(pyes_file)
    yield "p(yes) d3-flash", pyes_rows, np.array([row["score"] for row in pyes_rows], dtype=np.float64)

    encoder_fits = {
        "Qwen ViT patch_mean": RUN / "fit-qwen35-vit",
        "V-JEPA 2 patch_mean": RUN / "fit-vjepa2-vitl",
        "Qwen ViT centered full": RUN / "analysis-centered/fit-qwen35-vit",
        "V-JEPA 2 centered full": RUN / "analysis-centered/fit-vjepa2-vitl",
    }
    for label, fit_dir in encoder_fits.items():
        for method in ("mil", "holmes"):
            yield f"{label} {method}", *probe_scores(fit_dir, method, "patch_mean")

    llm_fit = MAIN / "probe_ucf_short_v1_f8/run-2/fit-C"
    yield "d3-flash LLM last holmes", *probe_scores(llm_fit, "holmes", "last")


def weighted_auc(values: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    """Frame-weighted AUC as in encoder-compare (positive / negative = event / other frames per window)."""
    return _weighted_curve(values, positive[None], negative[None])[0][0]


def smooth_per_video(values: np.ndarray, videos: list[np.ndarray], smoother: str) -> np.ndarray:
    """Apply the smoother to each video's windows separately; videos holds the row indices of each video."""
    out = values.copy()
    for rows in videos:
        out[rows] = smooth(values[rows], smoother)
    return out


def per_video_aucs(values, positive, negative, videos, is_anomaly) -> np.ndarray:
    """In-video AUC of each anomaly video that has both event and other frames."""
    aucs = []
    for rows in videos:
        if not is_anomaly[rows[0]] or positive[rows].sum() == 0 or negative[rows].sum() == 0:
            continue
        aucs.append(weighted_auc(values[rows], positive[rows], negative[rows]))
    return np.array(aucs)


def main() -> None:
    items = {item.name: item for item in DATASETS.build(load_encoder_config(RUN / "config.yaml").test).items()}
    print(f"{'source':34s} {'smoother':9s}  all    vs_normal in_video  per-video median  below 0.5")
    for name, meta, values in sources():
        positive, negative = _window_counts(meta, items)
        video_names = np.array([row["video"] for row in meta])
        is_anomaly = np.array([not video.startswith("Normal") for video in video_names])
        videos = [np.flatnonzero(video_names == video) for video in dict.fromkeys(video_names)]
        for smoother in SMOOTHERS:
            smoothed = smooth_per_video(values, videos, smoother)
            overall = weighted_auc(smoothed, positive, negative)
            vs_normal = weighted_auc(smoothed, positive, negative * ~is_anomaly)
            in_video = weighted_auc(smoothed, positive, negative * is_anomaly)
            per_video = per_video_aucs(smoothed, positive, negative, videos, is_anomaly)
            below = int((per_video < 0.5).sum())
            print(f"{name:34s} {smoother:9s} {overall:.3f}  {vs_normal:.3f}     {in_video:.3f}     "
                  f"{np.median(per_video):.3f}             {below}/{len(per_video)}")
        print()


if __name__ == "__main__":
    main()
