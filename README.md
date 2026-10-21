# system-one-streaming-vlm

Streaming anomaly detection with System One models: one forward pass per video window, no text generation.
The package `s1stream` runs an experiment from one YAML config and evaluates latency and window classification.

## How an experiment works

```
config.yaml ─► s1stream run (GPU)
                 dataset ─► windows ─► V1 input form ─► V2 backbone ─► V3 answer head ─► scores.jsonl
               s1stream report (offline, no model)
                 scores.jsonl ─► thresholds (threshold rule) ─► metrics ─► report.json
```

- The model runs once per config. Metrics run later on the saved scores and times.
- Each slot is a plugin that registers under a name. A config names the plugin and its parameters.

| Slot | Plugins now | File |
|---|---|---|
| dataset | `ucf_crime` | `src/s1stream/datasets/ucf_crime.py` |
| input form (V1) | `frames` | `src/s1stream/input_forms/frames.py` |
| backbone (V2) | `jev_omni`, `clef_flash`, `imajev`, `gemma_generate`, `d3` | `src/s1stream/backbones/<name>.py` |
| answer head (V3) | `option_prob`, `option_logits`, `generated_answer` | `src/s1stream/heads/<name>.py` |
| decision rule | `threshold` | `src/s1stream/rules/threshold.py` |

The slot protocols are in `src/s1stream/slots.py`. The shared data types are in `src/s1stream/types.py`.

## Setup

Requirements: Linux, an NVIDIA GPU with CUDA 13 (developed on a GB10), [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Taide2026/system-one-streaming-vlm.git
cd system-one-streaming-vlm
uv sync --managed-python
uv run pytest -q          # 99 tests, CPU only, no model download
```

`--managed-python` installs a uv-managed Python. Triton compiles a C helper at run time and needs `Python.h`.
A uv-managed Python has it. The system Python has it only when `python3.12-dev` is installed.

Torch comes from the PyTorch cu130 index. Transformers is pinned to 5.17.0, the version the Jev-Omni checkpoint was tested with.

## Assets (models and datasets)

Models and datasets are never in git. `assets.yaml` lists each asset with its source, exact revision and file SHA-256.
A config names an asset (`asset: jev-omni`), never a path.

1. Pick one folder per machine for assets and set `S1_ASSETS` (default: `./assets`).
2. Asset `<name>` lives at `$S1_ASSETS/<name>`.
3. Download and check: `uv run s1stream fetch jev-omni ucf-crime`.
4. Check only: `uv run s1stream fetch --check jev-omni`.

A config names its assets in its `asset:` fields. Download only the assets of the configs you run.
When `assets.yaml` gets a new asset after `git pull`, run `fetch` for that name only.

| Asset | Source | Size | Used by |
|---|---|---|---|
| `ucf-crime` | UCF-Crime zip (Dropbox); `fetch` unzips it and builds the layout below | 98 GB | all configs |
| `jev-omni` | `akhilaaa3/Jev-Omni` @ `5addda8` | 24 GB | `jev_omni_*` |
| `clef-flash` | `Cloudflare/clef-flash` @ `8b2e5fd` | 19 GB | `clef_flash_*` |
| `imajev-4b` | `mohit67890/imajev-4b` @ `27d4e54` (LoRA + readout; needs `qwen3.5-4b`) | 1.0 GB | `imajev_4b_*` |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` @ `851bf6e` (base model of `imajev-4b`) | 9.3 GB | `imajev_4b_*` |
| `gemma-4-e4b` | `google/gemma-4-E4B-it` @ `ee0ef60` | 16 GB | `gemma4_e4b_*` |
| `d3` | `vllm-sr/d3` @ `225be8b` | 52 GB | `d3_ucf_*` |
| `d3-flash` | `vllm-sr/d3-flash` @ `8b35c85` | 17 GB | `d3_flash_*` |
| `d3-mini` | `vllm-sr/d3-mini` @ `97dd054` | 9.1 GB | `d3_mini_*` |
| `d3-nano` | `vllm-sr/d3-nano` @ `aeb70d4` | 4.5 GB | `d3_nano_*` |
| `d3-lite` | `vllm-sr/d3-lite` @ `9c10703` | 1.7 GB | `d3_lite_*` |
| `d3-edge` | `vllm-sr/d3-edge` @ `cab71e6` | 1.2 GB | `d3_edge_*` |
| `laya` | `convaiinnovations/laya` @ `7b928d8` | 2.3 GB | none yet |
| `clm-8b` | `Contrastive-LM/CLM-v0.1-8B` @ `e939398` (heads only; needs Qwen3-8B) | 73 MB | none yet |

Sizes are the full Hugging Face snapshot at that revision. `fetch` downloads the full snapshot.

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

### UCF-Crime from ucf-clips

`scripts/run.sh assets` builds `ucf-crime` without the 98 GB zip. It uses an extracted copy of the
Hugging Face dataset `bear7011/ucf-clips` (`labels.csv`, `short/`, `medium/`, `long/`), set by `UCF_CLIPS`.

1. It downloads the annotation file from the UCF-Crime authors' repository and checks its SHA-256 against `assets.yaml`.
2. It links every video of `labels.csv` into `videos/<Class>/` (symlinks, no copies).
3. It also runs `uv run s1stream fetch jev-omni`.

This layout has no `splits/` and no `ReadMe-Anomaly-Detection.txt`. The `ucf_crime` dataset does not read them,
but `fetch --check ucf-crime` fails on this layout.

**Do not run it on a machine that has `ucf-crime` from the zip.** Before it makes a link, it deletes the file
of the same name in `videos/`. On a zip layout, that file is the real video.

## Run an experiment

```bash
export S1_ASSETS=/path/to/assets
free -g                                                     # jev_omni needs about 23 GB of GPU memory
timeout 30m uv run s1stream run configs/jev_omni_ucf20.yaml --out results/smoke/try --limit-windows 5
timeout 6h uv run s1stream run configs/jev_omni_ucf20.yaml  # writes results/jev_omni_ucf20/<YYYYmmdd-HHMMSS>/
uv run s1stream report results/jev_omni_ucf20/<run>
```

- `run` refuses an output folder that already exists. A rerun gets a new folder.
- `--limit-windows N` stops after N windows. Use it for a smoke test before a real run.
- The backbone caps its GPU memory with `memory_cap_gb` (default 40).
- A full run takes a while: start it inside tmux.

### scripts/run.sh

`scripts/run.sh` wraps the same commands. Run it without arguments to print its usage.

| Command | What it does |
|---|---|
| `scripts/run.sh setup` | installs uv when missing, `uv sync --managed-python`, checks `Python.h`, runs the CPU tests |
| `scripts/run.sh assets` | `fetch jev-omni`, then builds `ucf-crime` from ucf-clips (see above) |
| `scripts/run.sh smoke [config]` | 5 windows into `results/smoke/<time>`, then prints a short report |
| `scripts/run.sh run [config]` | full run into `results/<config name>/<time>`, then prints a short report |
| `scripts/run.sh report <run dir>` | `s1stream report` and the short report, no GPU |
| `scripts/run.sh all [config]` | `setup`, `assets`, `smoke`, `run` |

- The default config is `configs/jev_omni_ucf20.yaml`.
- `S1_ASSETS` defaults to `~/s1-assets` in this script (not `./assets`). Set it when your assets are elsewhere.
- `UCF_CLIPS` defaults to `~/datasets/ucf-clips`.
- The script adds no `timeout`. It prints `free -g` but does not stop when memory is low.

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

The score of a window is the probability of the `positive` option. `positive` may also list several options; the
score is then the sum of their probabilities (e.g. every anomaly class of a many-class question, = 1 − p(Normal)).

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

| Field | Definition | Unit |
|---|---|---|
| `latency.ms_per_frame` | Σ L ÷ total sampled frames | ms/frame |
| `latency.p50_ms`, `p95_ms`, `p99_ms` | percentiles of the per-window latency L | ms |
| `latency.rtf_p95` | P95(L) ÷ budget; ≤ 1 meets the budget | ratio |
| `classification[].accuracy` | correct windows ÷ all windows, per threshold | fraction |
| `classification[].recall`, `fnr` | TP ÷ (TP + FN); FN ÷ (TP + FN), per threshold | fraction |
| `classification[].specificity`, `balanced_accuracy` | TN ÷ (TN + FP); (recall + specificity) ÷ 2, per threshold | fraction |
| `decision` | `classification` at 0.5: the model's own answer | — |
| `auc.window`, `auc.frame` | ROC-AUC on window scores; on frames, each frame taking its window's score (UCF-Crime convention) | 0–1 |

- L is the `seconds` field of `scores.jsonl`: frame read + preprocessing + backbone + answer head, with CUDA synchronized at both ends.
- The first window is warm-up. Every latency field leaves it out; `latency.first_window_ms` shows it.
- The budget is `window.stride_s` (`latency.budget_ms`).
- Positive = abnormal. A window is abnormal when it overlaps an event by at least 1 frame. A window is predicted abnormal when its score ≥ the threshold.
- The thresholds are the settings of the `threshold` rules in the config, or 0.5 when there is none.
- `report.json` also has `windows`, `videos`, `frames_per_window`, `peak_gpu_gb` and `load_s`.

## Results

| Run | Windows | ms/frame | Latency P50 / P95 / P99 | RTF P95 | Threshold 0.1: accuracy / recall / FNR |
|---|---|---|---|---|---|
| `results/jev_omni_ucf20/20261009-feasibility` | 364 (20 videos) | 197 | 1573 / 1660 / 1692 ms | 0.83 | 79.7% / 64.2% / 35.8% |

This run is a feasibility check on 20 short test videos (11 events), not a benchmark.

UCF-Crime short (all 117 test videos under 50 s: 1684 windows of 2 s, 8 or 16 frames each) has one run folder per
model and frame count under `results/<model>_ucf_short_f{8,16}/`, from `configs/<model>_ucf_short_f{8,16}.yaml`:

| Backbone | Configs (`<model>`) |
|---|---|
| `jev_omni` | `jev_omni` |
| `clef_flash` | `clef_flash` |
| `imajev` | `imajev_4b` |
| `gemma_generate` (Gemma 4 E4B) | `gemma4_e4b_constrained` (structured prompt, xgrammar), `gemma4_e4b_pure_prob` (question only, xgrammar), `gemma4_e4b_pure` (question only, free decoding) |
| `gemma_generate` (Gemma 4 E4B full fine-tunes, THChou1220) | `gemma4_e4b_k54k_fall_{constrained,pure_prob}`, `gemma4_e4b_k417k_{constrained,pure_prob}` |
| `d3` (Decision 3.0) | `d3`, `d3_flash`, `d3_mini`, `d3_nano`, `d3_lite`, `d3_edge` |

`results/<model>_ucf_short_c14_f8/` (`configs/<model>_ucf_short_c14_f8.yaml`) are the same 8-frame runs with a
14-class prompt: the 13 UCF-Crime anomaly classes + Normal as options, score 1 − p(Normal). Models:
`gemma4_e4b_constrained`, `clef_flash`, `imajev_4b`, `jev_omni` and the six `d3` sizes.

All 290 UCF-Crime test videos with frame annotations (140 anomaly + 150 normal, 10.3 h: 18390 windows of 2 s, 8
frames) have one Yes / No run per model under `results/<model>_ucf_test_f8/` (`configs/<model>_ucf_test_f8.yaml`), for
`d3`, `d3_flash`, `d3_nano`, `clef_flash` and `gemma4_e4b_constrained`.

The 14-class prompt on all 290 reuses the short runs: `configs/<model>_ucf_long_c14_f8.yaml` runs only the 173 videos
above 49.9 s (`dataset.min_duration_s`), then `uv run python scripts/merge-c14-290.py <model>` joins both into
`results/<model>_ucf_test_c14_f8/` (it checks that the windows equal the Yes / No run's) and writes the report.

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
scripts/run.sh              setup, assets, smoke and full runs on the GPU machine
src/s1stream/               the package (slots, plugins, runner, report, metrics, assets, cli)
tests/                      CPU tests
results/                    run folders
```

## Commit messages

Conventional Commits: `<type>(<scope>): <subject>`, for example `fix(runner): write relative manifest paths`.
