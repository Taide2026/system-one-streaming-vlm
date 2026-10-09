# system-one-streaming-vlm

Streaming anomaly detection with System One models: one forward pass per video window, no text generation.
The package `s1stream` runs an experiment from one YAML config and evaluates accuracy, calibration, latency and alarm delay.

## How an experiment works

```
config.yaml ─► s1stream run (GPU)
                 dataset ─► windows ─► V1 input form ─► V2 backbone ─► V3 answer head ─► scores.jsonl
               s1stream report (offline, no model)
                 scores.jsonl ─► decision rules (threshold, …) ─► metrics ─► report.json
```

- The model runs once per config. Decision rules and metrics run later on the saved scores.
- Each slot is a plugin that registers under a name. A config names the plugin and its parameters.

| Slot | Plugins now | File |
|---|---|---|
| dataset | `ucf_crime` | `src/s1stream/datasets/ucf_crime.py` |
| input form (V1) | `frames` | `src/s1stream/input_forms/frames.py` |
| backbone (V2) | `jev_omni` | `src/s1stream/backbones/jev_omni.py` |
| answer head (V3) | `option_prob` | `src/s1stream/heads/option_prob.py` |
| decision rule | `threshold` | `src/s1stream/rules/threshold.py` |

The slot protocols are in `src/s1stream/slots.py`. The shared data types are in `src/s1stream/types.py`.

## Setup

Requirements: Linux, an NVIDIA GPU with CUDA 13 (developed on a GB10), [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Taide2026/system-one-streaming-vlm.git
cd system-one-streaming-vlm
uv sync
uv run pytest -q          # 53 tests, CPU only, no model download
```

Torch comes from the PyTorch cu130 index. Transformers is pinned to 5.17.0, the version the Jev-Omni checkpoint was tested with.

## Assets (models and datasets)

Models and datasets are never in git. `assets.yaml` lists each asset with its source, exact revision and file SHA-256.
A config names an asset (`asset: jev-omni`), never a path.

1. Pick one folder per machine for assets and set `S1_ASSETS` (default: `./assets`).
2. Asset `<name>` lives at `$S1_ASSETS/<name>`.
3. Download and check: `uv run s1stream fetch jev-omni ucf-crime`.
4. Check only: `uv run s1stream fetch --check jev-omni`.

| Asset | Source | Size |
|---|---|---|
| `ucf-crime` | UCF-Crime zip (Dropbox); `fetch` unzips it and builds the layout below | 98 GB |
| `jev-omni` | `akhilaaa3/Jev-Omni` @ `5addda8` | 24 GB |
| `laya` | `convaiinnovations/laya` @ `7b928d8` | 2.3 GB |
| `clm-8b` | `Contrastive-LM/CLM-v0.1-8B` @ `e939398` (heads only; needs Qwen3-8B) | 73 MB |

You may already have an asset somewhere else. Then put a link in `$S1_ASSETS` instead of downloading again:
`ln -s /data/ucf $S1_ASSETS/ucf-crime`, then run `fetch --check`.

The download paths of `fetch` (Hugging Face and the zip) have no automatic tests. The hash check and the UCF-Crime layout step do.

### UCF-Crime layout

Every path in every split file opens as `videos/<path>`.

```
ucf-crime/
  videos/<Class>/                         950 anomaly videos, 13 classes
  videos/Training_Normal_Videos_Anomaly/  800
  videos/Testing_Normal_Videos_Anomaly/   150
  videos/Normal_Videos_event/             50 (copies of test normal videos)
  annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
  splits/Anomaly_Train.txt, splits/Anomaly_Test.txt, splits/Action_Regnition_splits/
```

The annotation line `Arrest001_x264.mp4  Arrest  1185  1485  -1  -1` gives the start and end frame of up to 2 events (-1 = none).
All 290 test videos are 30 fps, 320×240. Only the test videos have event times; no video has a text description.

## Run an experiment

```bash
export S1_ASSETS=/path/to/assets
free -g                                                     # the run needs about 23 GB of GPU memory
uv run s1stream run configs/jev_omni_ucf20.yaml --out results/smoke/try --limit-windows 5
uv run s1stream run configs/jev_omni_ucf20.yaml             # writes results/jev_omni_ucf20/<YYYYmmdd-HHMMSS>/
uv run s1stream report results/jev_omni_ucf20/<run>
```

- `run` refuses an output folder that already exists. A rerun gets a new folder.
- `--limit-windows N` stops after N windows. Use it for a smoke test before a real run.
- The backbone caps its GPU memory with `memory_cap_gb` (default 40).

### Config

```yaml
name: jev_omni_ucf20
dataset:     {name: ucf_crime, asset: ucf-crime, n_anomaly: 10, n_normal: 10, max_duration_s: 60, seed: 20261009}
window:      {length_s: 2.0, stride_s: 2.0, frames: 8}
question:    {state: "...", text: "...", options: ["Yes", "No"], positive: "Yes"}
input_form:  {name: frames}
backbone:    {name: jev_omni, asset: jev-omni, memory_cap_gb: 40}
answer_head: {name: option_prob, asset: jev-omni}
decision_rules:
  - {name: threshold, start: 0.0, stop: 1.0, step: 0.01}
```

The score of a window is the probability of the `positive` option.

### Run folder

| File | Contents |
|---|---|
| `config.yaml` | the config as read |
| `manifest.json` | the selected videos; paths relative to the dataset asset folder |
| `scores.jsonl` | one line per window: video, window, start_s, end_s, label, score, probabilities, seconds |
| `run.json` | free memory, peak GPU memory, load time, versions, git commit, asset revisions and hashes |
| `report.json` | the metrics below |

Run folders under `results/<config name>/` go into git. `results/smoke/` does not.

## Metrics

| Group | Metrics |
|---|---|
| Accuracy | window AUROC, frame-level AUC; per threshold: accuracy, precision, recall, F1 (window level); `majority_accuracy` = accuracy of marking every window as the larger class |
| Calibration | ECE, Brier score |
| Latency | seconds per window (first, mean, median, p95), real-time factor = mean ÷ stride, peak GPU memory, load time |
| Alarms (per decision rule setting) | detection rate, mean and median detection delay, false alarms per hour |

- A window is abnormal when it overlaps an event by at least 1 frame.
- An event is detected when an alarm window overlaps it. The delay = that window's end − the event start.
- A false alarm is an alarm on a window with label 0.

## Results

| Run | Windows | Time per window | Window AUROC | Threshold 0.1 |
|---|---|---|---|---|
| `results/jev_omni_ucf20/20261009-feasibility` | 364 (20 videos) | 1.58 s (real-time factor 0.79) | 0.826 | 82% detected, median delay 2.0 s, 318 false alarms per hour |

This run is a feasibility check on 20 short test videos (11 events), not a benchmark.

## Add a plugin

1. Write a class that implements one protocol of `src/s1stream/slots.py`.
2. Register it: `@BACKBONES.register("my_model")` (or `DATASETS`, `INPUT_FORMS`, `ANSWER_HEADS`, `DECISION_RULES`).
3. Import its module in `src/s1stream/plugins.py`.
4. Add its asset to `assets.yaml` when it needs files.
5. Write CPU tests with a tiny fake model. Real models run only in `s1stream run`.

## Repository layout

```
assets.yaml                 asset lock file
configs/                    one YAML file per experiment
src/s1stream/               the package (slots, plugins, runner, report, metrics, assets, cli)
tests/                      CPU tests
results/                    run folders
.specs/changes/             proposals and unit tables of each change
```

## Commit messages

Conventional Commits: `<type>(<scope>): <subject>`, for example `fix(runner): write relative manifest paths`.
