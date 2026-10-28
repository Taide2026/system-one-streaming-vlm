"""Read-out-level fine-tuning (stage 1, 2026-10-24): `s1stream train <config.yaml>`.

The recipe (deck p. 32, paper Sec. 4.2): the vision tower frozen; a LoRA on every language-model projection of the
Qwen3.5 backbone (attention q k v o, MLP gate up down, Gated DeltaNet in_proj_qkv / in_proj_z / out_proj); the model's
own read-out trained in float32 (Clef: the joint schema head; d3: the 255-code linear read-out); the loss a soft
cross-entropy over the question's options with the window's coverage as target; one epoch, AdamW, warm-up then cosine,
bf16 autocast, gradient checkpointing; afterwards one temperature fitted on the held-out training videos. The inference
path does not change: the backbone loads the adapter (merged) and the read-out from the run folder
(backbone: {adapter: <run folder>}). The prompt mask stays the release's (Clef causal, d3 noncausal: its PromptEncoder).

Config (configs/ft_ucf_v1_<model>.yaml):
  name, data: {hivau_json, asset: ucf-crime, n_abnormal, n_normal, n_val, holdout, seed, hard_labels, hard_negative_share (ablations),
  videos_dir + flat: true for XD-Violence (root/train-videos/<name>.mp4)},
  window: {length_s,
  stride_s, frames}, question: {prompt: v1}, backbone: {name: clef_flash | d3, asset}, lora: {r, alpha, dropout},
  optim: {lr, head_lr, warmup, batch, accumulate, epochs, workers, grad_checkpoint}
  lora.r = 0: a linear probe, the backbone frozen and only the read-out trained (d3, Clef); no adapter/ is written.

Run folder (--out, default results/<machine>/<name>/<time>):
  config.yaml, records.jsonl, val.jsonl, train.log (one JSON line per optimizer step), adapter/ (peft),
  head.safetensors (Clef) or readout.safetensors (d3), calibration.json {temperature, val_nll_before, val_nll_after,
  val_window_auc}, run.json.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from s1stream.assets import resolve
from s1stream.config import ConfigError
from s1stream.input_forms.frames import read_frames
from s1stream.traindata import Record, read_hivau, sample_records, write_records
from s1stream.types import Question

# Module names with or without a leading path (Clef: model.language_model.layers.N; d3: language_model.layers.N).
LORA_TARGETS = (r"^(.*\.)?language_model\.layers\.\d+\..*\."
                r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)$")


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """-sum_k target_k log softmax(logits)_k, averaged over the batch. logits, target: (batch, options)."""
    return -(target * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def targets(records: list[Record], option_ids: tuple[str, ...], positive: str) -> torch.Tensor:
    """(batch, options) soft targets: y on the positive option, 1 - y on the other one (two options only)."""
    if len(option_ids) != 2 or positive not in option_ids:
        raise ConfigError("The fine-tune needs a two-option question whose positive is one of the options")
    out = torch.zeros(len(records), 2)
    pos = option_ids.index(positive)
    for i, record in enumerate(records):
        out[i, pos] = record.y
        out[i, 1 - pos] = 1.0 - record.y
    return out


def load_train_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    for key in ("name", "data", "window", "question", "backbone", "lora", "optim"):
        if key not in data:
            raise ConfigError(f"Missing section: {key}")
    from s1stream.prompt import PROMPTS

    prompt = data["question"]["prompt"]
    if prompt not in PROMPTS:
        raise ConfigError(f"Unknown question.prompt {prompt!r}")
    data["question"] = {"prompt": prompt, **PROMPTS[prompt]}
    return data


def question_of(data: dict[str, Any]) -> Question:
    q = data["question"]
    positive = q["positive"]
    return Question(state=q["state"], text=q["text"], options=tuple(q["options"]),
                    positive=tuple(positive) if isinstance(positive, list) else positive)


def lr_at(step: int, total: int, warmup: int) -> float:
    """Linear warm-up to 1, then cosine to 0.1."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def fit_temperature(logits: torch.Tensor, target: torch.Tensor) -> float:
    """The T > 0 that minimises the soft cross-entropy of logits / T (golden-section search on log T)."""
    def nll(log_t: float) -> float:
        return soft_cross_entropy(logits / math.exp(log_t), target).item()

    lo, hi = math.log(0.1), math.log(10.0)
    phi = (math.sqrt(5) - 1) / 2
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = nll(a), nll(b)
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = nll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = nll(b)
    return math.exp((lo + hi) / 2)


def window_auc(logits: torch.Tensor, target: torch.Tensor, pos: int) -> float | None:
    """Window-level AUC of p(positive) against y >= 0.5, or None when one class is missing."""
    from s1stream.metrics import auc as auc_fn

    scores = logits.softmax(dim=-1)[:, pos].tolist()
    labels = [int(y >= 0.5) for y in target[:, pos].tolist()]
    if len(set(labels)) < 2:
        return None
    return auc_fn(scores, labels)


def lora_wrap(backbone: torch.nn.Module, r: int, alpha: int, dropout: float, grad_checkpoint: bool):
    """Freeze backbone, enable gradient checkpointing, and inject a LoRA on LORA_TARGETS. Returns the peft model
    (the backbone's modules are modified in place, so references to its layers stay valid)."""
    backbone.requires_grad_(False)
    if r <= 0:  # linear probe (2026-10-28): no adapter, the read-out alone trains on frozen features
        return backbone
    from peft import LoraConfig, get_peft_model

    if grad_checkpoint:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.enable_input_require_grads()
    return get_peft_model(backbone, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                                               target_modules=LORA_TARGETS))


# ---------------------------------------------------------------------------------------------------------------
# Trainees: one per backbone family. encode (worker, CPU) -> collate -> forward (GPU, returns (batch, options)
# logits in option order given by option_ids) -> save.


class ClefTrainee:
    """Clef-Flash: the release's ClefModel with a LoRA on the backbone and the joint schema head in float32."""

    name = "clef_flash"

    def __init__(self, path: Path, device: torch.device, question: Question, lora: dict, grad_checkpoint: bool):
        from s1stream.backbones.clef_flash import ClefFlash, build_record

        self.build_record = build_record
        wrapper = ClefFlash(path, memory_cap_gb=100.0, device=str(device))
        self.clef, self.code, self.processor = wrapper.model, wrapper.code, wrapper.processor
        self.question, self.device = question, device
        self.clef.language_model = lora_wrap(self.clef.language_model, int(lora["r"]), int(lora["alpha"]),
                                             float(lora.get("dropout", 0.0)), grad_checkpoint)
        self.clef.head.float().requires_grad_(True)
        # Option ids in the release's order (it sorts choice options by id); fixed by the question, so computed
        # here rather than in collate, which runs in the loader's worker processes.
        probe = self.code.encode_record(self.processor.tokenizer, self.build_record(question, []))
        self.option_ids: tuple[str, ...] = tuple(probe.questions[0].option_ids)

    def encode(self, images: list) -> Any:
        return self.code.encode_record(self.processor.tokenizer, self.build_record(self.question, images),
                                       processor=self.processor)

    def collate(self, encoded: list) -> dict[str, Any]:
        return self.code.collate_records(encoded, self.processor.tokenizer.pad_token_id, torch.device("cpu"))

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        base = self.clef.language_model.get_base_model()
        media = {k: v.to(self.device) for k, v in (batch.get("media") or {}).items()}
        input_ids, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)
        text_model = base.model
        if not media and hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        outputs = text_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True,
                             **media)
        per_record = self.clef.head(outputs.last_hidden_state.float(), input_ids, attention_mask, batch["records"],
                                    base.get_output_embeddings().weight.float())
        return torch.stack([record_logits[0] for record_logits in per_record])

    def parameters(self) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        lora = [p for p in self.clef.language_model.parameters() if p.requires_grad]
        return lora, list(self.clef.head.parameters())

    def train(self, mode: bool) -> None:
        self.clef.train(mode)

    def save(self, out_dir: Path) -> None:
        from safetensors.torch import save_file

        if hasattr(self.clef.language_model, "save_pretrained") and hasattr(self.clef.language_model, "peft_config"):
            self.clef.language_model.save_pretrained(out_dir / "adapter")  # not for a probe (lora r 0)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.clef.head.state_dict().items()},
                  out_dir / "head.safetensors")


class D3Trainee:
    """Decision 3.0 (d3-nano): the release runtime's image path run with gradients: image_embeddings -> the
    noncausal PromptEncoder (or the stock forward for a causal checkpoint) -> last position -> the 255-code
    read-out as a float32 parameter. Option k of the question is code k (question.options order)."""

    name = "d3"

    def __init__(self, path: Path, device: torch.device, question: Question, lora: dict, grad_checkpoint: bool):
        from s1stream.backbones.d3 import build_questions, load_release_code, runtime_device

        self.code = load_release_code(path)
        self.runtime = self.code.D3.from_pretrained(path, device=runtime_device(str(device)), local_files_only=True)
        if self.runtime.processor is None:
            raise ConfigError(f"d3 at {path}: no image processor ({self.runtime.image_unavailable})")
        self.question, self.device = question, device
        normalized = self.code.normalize_question(build_questions(question)["answer"])
        self.rendered = normalized.rendered
        self.option_ids = tuple(question.options)
        self.peft = lora_wrap(self.runtime.backbone, int(lora["r"]), int(lora["alpha"]),
                              float(lora.get("dropout", 0.0)), grad_checkpoint)
        self.readout = torch.nn.Parameter(self.runtime.readout.detach().float().clone())
        self.runtime.fast = None

    def encode(self, images: list) -> Any:
        text = self.runtime.image_text(self.question.state, self.rendered, len(images))
        return text, images

    def collate(self, encoded: list) -> dict[str, Any]:
        texts = [text for text, _ in encoded]
        images = [image for _, imgs in encoded for image in imgs]
        return dict(self.runtime.processor(text=texts, images=images, padding=True, return_tensors="pt"))

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        with self.code.sdpa_backends(self.device):
            if self.runtime.encoder is not None:
                embeds, mrope = self.runtime.image_embeddings(inputs)
                hidden = self.runtime.encoder(embeds, inputs["attention_mask"], mrope)[:, -1]
            else:
                hidden = self.runtime.backbone(**inputs, use_cache=False).last_hidden_state[:, -1]
        logits = hidden.float() @ self.readout.T
        return logits[:, :len(self.option_ids)]

    def parameters(self) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        return [p for p in self.peft.parameters() if p.requires_grad], [self.readout]

    def train(self, mode: bool) -> None:
        self.runtime.backbone.train(mode)

    def save(self, out_dir: Path) -> None:
        from safetensors.torch import save_file

        if hasattr(self.peft, "peft_config"):  # a probe (lora r 0) is the plain backbone: no adapter to save
            self.peft.save_pretrained(out_dir / "adapter")
        save_file({"weight": self.readout.detach().cpu().contiguous()}, out_dir / "readout.safetensors")


class QwenTrainee:
    """Qwen3.5 instruction model without decision training (the control): the backbone qwen_first_token's prompt,
    the next-token logits at the answer position gathered at each option's first token; LoRA only, no head."""

    name = "qwen_first_token"

    def __init__(self, path: Path, device: torch.device, question: Question, lora: dict, grad_checkpoint: bool):
        from s1stream.backbones.qwen_first_token import ANSWER_TAIL, QwenFirstToken, build_prompt, first_tokens

        wrapper = QwenFirstToken(path, memory_cap_gb=100.0, device=str(device))
        self.processor, self.device, self.question = wrapper.processor, device, question
        self.option_ids = tuple(question.options)
        self.token_ids = torch.tensor(first_tokens(self.processor.tokenizer, question.options), device=device)
        self.peft = lora_wrap(wrapper.model, int(lora["r"]), int(lora["alpha"]), float(lora.get("dropout", 0.0)),
                              grad_checkpoint)
        self.build_prompt, self.answer_tail = build_prompt, ANSWER_TAIL

    def encode(self, images: list) -> Any:
        messages = [{"role": "user", "content": [{"type": "image"}] * len(images)
                     + [{"type": "text", "text": self.build_prompt(self.question)}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                  enable_thinking=False)
        if not text.endswith(self.answer_tail):
            raise RuntimeError("Unexpected Qwen non-thinking template boundary")
        return text, images

    def collate(self, encoded: list) -> dict[str, Any]:
        self.processor.tokenizer.padding_side = "left"
        texts = [text for text, _ in encoded]
        images = [image for _, imgs in encoded for image in imgs]
        return dict(self.processor(text=texts, images=images, padding=True, return_tensors="pt"))

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        logits = self.peft(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
        return logits.float()[:, self.token_ids]

    def parameters(self) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        return [p for p in self.peft.parameters() if p.requires_grad], []

    def train(self, mode: bool) -> None:
        self.peft.train(mode)

    def save(self, out_dir: Path) -> None:
        self.peft.save_pretrained(out_dir / "adapter")


TRAINEES = {"clef_flash": ClefTrainee, "d3": D3Trainee, "qwen_first_token": QwenTrainee}


class WindowDataset(torch.utils.data.Dataset):
    """Reads the frames of a record and encodes it for the trainee, in the worker."""

    def __init__(self, records: list[Record], encode) -> None:
        self.records, self.encode = records, encode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return self.encode(read_frames(Path(record.path), record.frame_indices)), record


def make_loader(records: list[Record], trainee, batch: int, workers: int, shuffle: bool):
    def collate(items):
        return trainee.collate([item[0] for item in items]), [item[1] for item in items]

    return torch.utils.data.DataLoader(WindowDataset(records, trainee.encode), batch_size=batch, shuffle=shuffle,
                                       num_workers=workers, collate_fn=collate,
                                       prefetch_factor=4 if workers else None)


@torch.no_grad()
def collect_logits(trainee, loader, device: torch.device, positive: str) -> tuple[torch.Tensor, torch.Tensor]:
    trainee.train(False)
    all_logits, all_targets = [], []
    for batch, records in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = trainee.forward(batch)
        all_logits.append(logits.float().cpu())
        all_targets.append(targets(records, trainee.option_ids, positive))
    return torch.cat(all_logits), torch.cat(all_targets)


def train(config_path: Path, out_dir: Path, limit_steps: int | None = None) -> Path:
    data = load_train_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    started = datetime.now(timezone.utc).isoformat()
    question = question_of(data)
    window, dspec, lora, optim = data["window"], data["data"], data["lora"], data["optim"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(dspec.get("seed", 0)))
    batch_size, accumulate = int(optim["batch"]), int(optim.get("accumulate", 1))
    workers = int(optim.get("workers", 4))

    # ---- records
    root = resolve(dspec["asset"])
    items = read_hivau(Path(os.path.expandvars(os.path.expanduser(dspec["hivau_json"]))), root,
                       str(dspec.get("videos_dir", "videos")), bool(dspec.get("flat", False)),
                       bool(dspec.get("skip_missing", False)))
    train_records, val_records = sample_records(
        items, float(window["length_s"]), float(window["stride_s"]), int(window["frames"]),
        int(dspec["n_abnormal"]), int(dspec["n_normal"]), int(dspec.get("n_val", 0)),
        float(dspec.get("holdout", 0.1)), int(dspec.get("seed", 0)),
        float(dspec.get("hard_negative_share", 0.0)))
    if dspec.get("hard_labels"):  # ablation: y >= 0.5 -> 1, else 0 (the soft coverage target is the default)
        from dataclasses import replace

        train_records = [replace(r, y=1.0 if r.y >= 0.5 else 0.0) for r in train_records]
        val_records = [replace(r, y=1.0 if r.y >= 0.5 else 0.0) for r in val_records]
    if limit_steps is not None:
        train_records = train_records[:limit_steps * batch_size * accumulate]
        val_records = val_records[:min(len(val_records), 64)]
    write_records(train_records, out_dir / "records.jsonl")
    write_records(val_records, out_dir / "val.jsonl")
    print(f"{len(items)} training videos, {len(train_records)} training windows, {len(val_records)} held-out windows",
          flush=True)

    # ---- model
    spec = data["backbone"]
    if spec["name"] not in TRAINEES:
        raise ConfigError(f"No trainee for backbone {spec['name']!r}: one of {', '.join(TRAINEES)}")
    path = resolve(spec["asset"]) if "asset" in spec else Path(spec["path"])
    load_start = time.perf_counter()
    trainee = TRAINEES[spec["name"]](path, device, question, lora, bool(optim.get("grad_checkpoint", True)))
    trainee.train(True)
    lora_params, head_params = trainee.parameters()
    if not lora_params and not head_params:
        raise ConfigError("Nothing to train: lora.r is 0 and this backbone has no read-out head (a probe needs d3 or Clef)")
    n_lora, n_head = sum(p.numel() for p in lora_params), sum(p.numel() for p in head_params)
    print(f"trainable: LoRA {n_lora / 1e6:.1f} M, read-out {n_head / 1e6:.1f} M; load {time.perf_counter() - load_start:.0f} s",
          flush=True)
    positive = question.positive if isinstance(question.positive, str) else question.positive[0]
    loader = make_loader(train_records, trainee, batch_size, workers, True)

    epochs = int(optim.get("epochs", 1))
    total_steps = math.ceil(len(loader) / accumulate) * epochs
    warmup = int(round(total_steps * float(optim.get("warmup", 0.03))))
    base_lr, head_lr = float(optim["lr"]), float(optim.get("head_lr", optim["lr"]))
    groups = [{"params": lora_params, "lr": base_lr}] if lora_params else []
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_at(s, total_steps, warmup))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- loop
    step, micro, running = 0, 0, 0.0
    log = (out_dir / "train.log").open("w")
    t0 = time.perf_counter()
    done = False
    for epoch in range(epochs):
        for batch, records in loader:
            target = targets(records, trainee.option_ids, positive).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = trainee.forward(batch)
            loss = soft_cross_entropy(logits, target) / accumulate
            loss.backward()
            running += loss.item()
            micro += 1
            if micro % accumulate:
                continue
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            elapsed = time.perf_counter() - t0
            entry = {"step": step, "epoch": epoch, "loss": running, "lr": scheduler.get_last_lr()[0],
                     "seconds": elapsed, "s_per_step": elapsed / step, "windows_seen": micro * batch_size,
                     "eta_min": (total_steps - step) * elapsed / step / 60}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step % 10 == 0 or step == 1:
                print(f"step {step}/{total_steps} loss {running:.4f} lr {entry['lr']:.2e} "
                      f"{entry['s_per_step']:.1f} s/step eta {entry['eta_min']:.0f} min", flush=True)
            running = 0.0
            if limit_steps is not None and step >= limit_steps:
                done = True
                break
        if done:
            break
    log.close()
    train_seconds = time.perf_counter() - t0

    # ---- save, then the temperature on the held-out training videos
    trainee.save(out_dir)
    calibration: dict[str, Any] = {"temperature": 1.0}
    if val_records:
        logits, target = collect_logits(trainee, make_loader(val_records, trainee, batch_size, workers, False),
                                        device, positive)
        pos = list(trainee.option_ids).index(positive)
        temperature = fit_temperature(logits, target)
        calibration = {"temperature": temperature,
                       "val_nll_before": soft_cross_entropy(logits, target).item(),
                       "val_nll_after": soft_cross_entropy(logits / temperature, target).item(),
                       "val_window_auc": window_auc(logits, target, pos), "val_windows": len(val_records)}
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=1))

    run = {"name": data["name"], "backbone": spec["name"],
           "machine": os.environ.get("S1_MACHINE") or os.uname().nodename,
           "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
           "started": started, "finished": datetime.now(timezone.utc).isoformat(),
           "train_seconds": train_seconds, "steps": step, "total_steps": total_steps,
           "train_windows": len(train_records), "val_windows": len(val_records), "limit_steps": limit_steps,
           "trainable": {"lora": n_lora, "readout": n_head},
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
           "calibration": calibration}
    (out_dir / "run.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(run, indent=1))
    return out_dir
