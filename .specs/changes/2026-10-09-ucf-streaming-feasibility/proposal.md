# ucf-streaming-feasibility: run Jev-Omni as a streaming anomaly detector on 20 UCF-Crime test videos

Status: done (2026-10-09)

## Problem

We do not know if Jev-Omni can run as a streaming System One model on the GB10.
We do not know its time per window, its peak memory, or its alarm delay.
A full run on the 290 UCF-Crime test videos (10.3 h of video) can take more than 10 h.
A small run on 20 videos answers the feasibility question in about 10 to 30 min.
The repository is empty, so this change also sets up the project.

## Decisions

1. The project uses uv. It pins `transformers==5.17.0`, the version in the Jev-Omni `requirements.txt`. It uses the same torch build as `streamdecide_pilot` (2.14.1+cu130).
2. The feasibility set has 20 UCF-Crime test videos: 10 anomaly and 10 normal. Each video is 60 s or shorter. A fixed seed (20261009) picks them. The list is stored in `data/feasibility_20.json` with the video path, duration, frame count and event frame ranges.
3. A window is 2 s long. The stride is 2 s, so windows do not overlap. Each window holds 8 frames, sampled evenly at 4 fps from the native 30 fps video. The decision time of a window is its end time.
4. The model is Jev-Omni at `/home/cctsai/Eric/VLM/ckpt/Jev-Omni`, revision `5addda8`. Our loader reads only that folder and never downloads. It takes a list of frames, not a video file.
5. Each window gets one question: "Is something abnormal happening, such as a crime, an accident, a fight, a fire or an explosion?" with the options "Yes" and "No". The state text is "Surveillance camera footage, 2 seconds." The abnormal score is the probability of "Yes".
6. A window is labeled abnormal when it overlaps an annotated event frame range by at least 1 frame.
7. The run writes one JSON line per window: video name, window start and end (s), abnormal score, label, and time per window (s). A separate JSON file holds the run data: free memory before the run, peak GPU memory, model load time, library versions and the git commit. Results go to `results/feasibility_20/<run name>/`. The run refuses to write into a folder that already exists.
8. The report computes the metrics of the dev doc section 12: window AUROC, frame-level AUC, ECE, Brier score, time per window, real-time factor, peak GPU memory, detection rate, detection delay and false alarms per hour. The alarm metrics use a threshold sweep from 0 to 1 and give a delay-vs-false-alarm table.
9. Resource guards: inference runs under `torch.inference_mode()`. The program sets `torch.cuda.set_per_process_memory_fraction` to a 40 GB cap. Every run uses `timeout`. A smoke run of 5 windows comes before the 20-video run. The smoke run starts only when `free -g` shows more than 40 GB available.
10. Unit tests run on the CPU with synthetic frames, synthetic scores and a tiny fake model. They do not load Jev-Omni.
11. The code is a framework with config and plugins (approach A, approved 2026-10-09). Each experiment is one YAML file in `configs/`. The file names the dataset, the window settings, the V1 input form, the V2 backbone, the V3 answer head and the decision rules.
12. Each slot is a Python protocol with a registry. A plugin registers under a name, and the config refers to that name. This change implements one plugin per slot: dataset `ucf_crime`, input form `frames`, backbone `jev_omni`, answer head `option_prob`, and decision rule `threshold`.
13. The GPU stage and the offline stage are separate commands. `s1stream run <config>` writes the window scores. `s1stream report <run dir>` applies the decision rules and computes the metrics. The report never loads a model.
14. A run folder holds `config.yaml` (a copy), `manifest.json`, `scores.jsonl`, `run.json` and `report.json`.
15. Models and datasets are assets. The git file `assets.yaml` lists each asset with its source, exact revision and file hashes. A config refers to an asset by name, never by an absolute path. Each machine sets one asset root in the environment variable `S1_ASSETS`; asset `<name>` lives at `$S1_ASSETS/<name>`. `s1stream fetch <name>` downloads an asset and checks its hashes. `s1stream fetch --check <name>` only checks. On this machine, `S1_ASSETS=/home/cctsai/Eric/VLM/assets`, which holds links to the existing folders. `run.json` records the revision and hashes of each asset it used. The manifest, `scores.jsonl` and `report.json` of each run go into git; models and videos never do.

## Reasons

- Decision 1: the model author tested 5.17.0. A different version can change the processor output and the probabilities.
- Decision 2: the 60 s limit keeps the run near 600 windows. The full test set has videos up to 3600 s.
- Decision 3: you chose the 2 s window and 8 frames. A 2 s stride halves the run time against a 1 s stride.
- Decision 4: the original loader calls `snapshot_download` and samples 16 frames from the whole file. Both conflict with streaming.
- Decision 5: one fixed question keeps the run comparable with later runs. Prompt tuning is a separate change.
- Decision 9: the GB10 has unified memory, and other processes use about 73 GB now.
- Decision 9 (measured): the smoke run on 2026-10-09 measured 1.42 s per window (after a 3.8 s first window), 22.6 GB peak GPU memory and 128 s load time. 2228 tokens go into each window.
- Decision 15: teammates must replay runs without access to this machine. A model upload to a remote repo is large and outward, so we commit the recipe to get each asset instead.
- Decision 13: the model runs once per config. Threshold, CUSUM and SPRT then cost no GPU time.

## Out of scope

- The other 270 test videos and the 1610 training videos.
- Plugins other than the five in decision 12: other models (Laya, CLM-8B, Gemma), other input forms (caption, labels), other answer heads (decode, probe) and other decision rules (CUSUM, SPRT). The slot protocols must allow them.
- The caption cache and the comparison table across runs.
- Prompt tuning, calibration fitting and training.
- Changes to `streamdecide_pilot/`, `/home/cctsai/Eric/VLM/ucf/` and `/home/cctsai/Eric/VLM/ckpt/`.
- A push to the GitHub remote.
