"""Read-out-level fine-tuning of Clef-Flash (stage 1, 2026-10-24): `s1stream train <config.yaml>`.

The recipe (deck p. 32, paper Sec. 4.2): the vision tower frozen; a LoRA on every language-model projection of the
Qwen3.5 backbone (attention q k v o, MLP gate up down, Gated DeltaNet in_proj_qkv / in_proj_z / out_proj); the joint
schema head trained in float32; the loss a soft cross-entropy over the question's options with the window's coverage
as target; one epoch, AdamW, warm-up then cosine, bf16 autocast, gradient checkpointing; afterwards one temperature
fitted on the held-out training videos. The inference path does not change: the backbone clef_flash loads the adapter
(merged) and the head from the run folder (backbone: {adapter: <run folder>}).

Config (configs/ft_ucf_v1_clef.yaml):
  name, data: {hivau_json, asset: ucf-crime, n_abnormal, n_normal, n_val, holdout, seed}, window: {length_s,
  stride_s, frames}, question: {prompt: v1}, backbone: {name: clef_flash, asset: clef-flash}, lora: {r, alpha,
  dropout}, optim: {lr, head_lr, warmup, batch, accumulate, epochs, workers, grad_checkpoint}

Run folder (--out, default results/<machine>/<name>/<time>):
  config.yaml, records.jsonl, val.jsonl, train.log (one JSON line per optimizer step), adapter/ (peft), head.safetensors
  (float32), calibration.json {temperature, val_nll_before, val_nll_after, val_window_auc}, run.json.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from s1stream.assets import resolve
from s1stream.backbones.clef_flash import ClefFlash, build_record
from s1stream.config import ConfigError
from s1stream.input_forms.frames import read_frames
from s1stream.traindata import Record, read_hivau, sample_records, write_records
from s1stream.types import Question

LORA_TARGETS = r".*\.language_model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)$"


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


class WindowDataset(torch.utils.data.Dataset):
    """Reads the frames of a record and encodes the Clef record (tokens + pixels) in the worker."""

    def __init__(self, records: list[Record], question: Question, code: Any, processor: Any) -> None:
        self.records, self.question, self.code, self.processor = records, question, code, processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        images = read_frames(Path(record.path), record.frame_indices)
        encoded = self.code.encode_record(self.processor.tokenizer, build_record(self.question, images),
                                          processor=self.processor)
        return encoded, record


def collate(batch, code: Any, pad_id: int):
    encoded = [item[0] for item in batch]
    records = [item[1] for item in batch]
    return code.collate_records(encoded, pad_id, torch.device("cpu")), records


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    out["input_ids"] = batch["input_ids"].to(device)
    out["attention_mask"] = batch["attention_mask"].to(device)
    out["media"] = {k: v.to(device) for k, v in batch["media"].items()}
    return out


def forward_logits(clef: Any, batch: dict[str, Any]) -> torch.Tensor:
    """The release's ClefModel.forward with the head run in float32: (batch, options) logits of the one question."""
    base = clef.language_model
    base = base.get_base_model() if hasattr(base, "get_base_model") else base
    media = batch.get("media") or {}
    text_model = base.model
    if not media and hasattr(text_model, "language_model"):
        text_model = text_model.language_model
    outputs = text_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False,
                         return_dict=True, **media)
    per_record = clef.head(outputs.last_hidden_state.float(), batch["input_ids"], batch["attention_mask"],
                           batch["records"], base.get_output_embeddings().weight.float())
    return torch.stack([record_logits[0] for record_logits in per_record])


def apply_lora(clef: Any, r: int, alpha: int, dropout: float, grad_checkpoint: bool) -> Any:
    """Wrap the backbone in a peft LoRA on LORA_TARGETS; the head becomes float32 and trainable; vision frozen."""
    from peft import LoraConfig, get_peft_model

    backbone = clef.language_model
    backbone.requires_grad_(False)
    if grad_checkpoint:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.enable_input_require_grads()
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=LORA_TARGETS)
    clef.language_model = get_peft_model(backbone, config)
    clef.head.float().requires_grad_(True)
    return clef


def trainable_parameters(clef: Any) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    lora = [p for n, p in clef.language_model.named_parameters() if p.requires_grad]
    head = [p for p in clef.head.parameters() if p.requires_grad]
    return lora, head


def lr_at(step: int, total: int, warmup: int) -> float:
    """Linear warm-up to 1, then cosine to 0.1."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def collect_logits(clef: Any, loader, device: torch.device, option_ids_of, positive: str
                   ) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    """(logits, targets, option ids) over a loader, for the temperature fit."""
    clef.eval()
    all_logits, all_targets, option_ids = [], [], ()
    for batch, records in loader:
        option_ids = option_ids_of(batch)
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = forward_logits(clef, batch)
        all_logits.append(logits.float().cpu())
        all_targets.append(targets(records, option_ids, positive))
    return torch.cat(all_logits), torch.cat(all_targets), option_ids


def fit_temperature(logits: torch.Tensor, target: torch.Tensor) -> float:
    """The T > 0 that minimises the soft cross-entropy of logits / T (golden-section search on log T)."""
    def nll(log_t: float) -> float:
        return soft_cross_entropy(logits / math.exp(log_t), target).item()

    lo, hi = math.log(0.2), math.log(5.0)
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


def train(config_path: Path, out_dir: Path, limit_steps: int | None = None) -> Path:
    data = load_train_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    started = datetime.now(timezone.utc).isoformat()
    question = question_of(data)
    window, dspec, lora, optim = data["window"], data["data"], data["lora"], data["optim"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(dspec.get("seed", 0)))

    # ---- records
    root = resolve(dspec["asset"])
    items = read_hivau(Path(dspec["hivau_json"]).expanduser(), root)
    train_records, val_records = sample_records(
        items, float(window["length_s"]), float(window["stride_s"]), int(window["frames"]),
        int(dspec["n_abnormal"]), int(dspec["n_normal"]), int(dspec.get("n_val", 0)),
        float(dspec.get("holdout", 0.1)), int(dspec.get("seed", 0)))
    if limit_steps is not None:
        train_records = train_records[:limit_steps * int(optim["batch"]) * int(optim.get("accumulate", 1))]
        val_records = val_records[:min(len(val_records), 64)]
    write_records(train_records, out_dir / "records.jsonl")
    write_records(val_records, out_dir / "val.jsonl")
    print(f"{len(items)} training videos, {len(train_records)} training windows, {len(val_records)} held-out windows")

    # ---- model
    backbone_spec = data["backbone"]
    path = resolve(backbone_spec["asset"]) if "asset" in backbone_spec else Path(backbone_spec["path"])
    load_start = time.perf_counter()
    wrapper = ClefFlash(path, memory_cap_gb=float(backbone_spec.get("memory_cap_gb", 100)), device=str(device))
    clef, code, processor = wrapper.model, wrapper.code, wrapper.processor
    clef = apply_lora(clef, int(lora["r"]), int(lora["alpha"]), float(lora.get("dropout", 0.0)),
                      bool(optim.get("grad_checkpoint", True)))
    clef.train()
    lora_params, head_params = trainable_parameters(clef)
    n_lora, n_head = sum(p.numel() for p in lora_params), sum(p.numel() for p in head_params)
    print(f"trainable: LoRA {n_lora / 1e6:.1f} M, head {n_head / 1e6:.1f} M; load {time.perf_counter() - load_start:.0f} s")

    pad_id = processor.tokenizer.pad_token_id
    make_loader = lambda records, shuffle: torch.utils.data.DataLoader(  # noqa: E731
        WindowDataset(records, question, code, processor), batch_size=int(optim["batch"]), shuffle=shuffle,
        num_workers=int(optim.get("workers", 4)), collate_fn=lambda b: collate(b, code, pad_id),
        persistent_workers=False, prefetch_factor=4 if int(optim.get("workers", 4)) else None)
    loader = make_loader(train_records, True)

    positive = question.positive if isinstance(question.positive, str) else question.positive[0]
    option_ids_of = lambda batch: batch["records"][0].questions[0].option_ids  # noqa: E731

    accumulate = int(optim.get("accumulate", 1))
    epochs = int(optim.get("epochs", 1))
    total_steps = math.ceil(len(loader) / accumulate) * epochs
    warmup = int(round(total_steps * float(optim.get("warmup", 0.03))))
    base_lr, head_lr = float(optim["lr"]), float(optim.get("head_lr", optim["lr"]))
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": base_lr}, {"params": head_params, "lr": head_lr}],
                                  weight_decay=0.0, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_at(s, total_steps, warmup))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- loop
    step, micro = 0, 0
    log = (out_dir / "train.log").open("w")
    t0 = time.perf_counter()
    running = 0.0
    for epoch in range(epochs):
        for batch, records in loader:
            batch = to_device(batch, device)
            target = targets(records, option_ids_of(batch), positive).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = forward_logits(clef, batch)
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
                     "seconds": elapsed, "s_per_step": elapsed / step,
                     "windows_seen": micro * int(optim["batch"]),
                     "eta_min": (total_steps - step) * elapsed / step / 60}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step % 10 == 0 or step == 1:
                print(f"step {step}/{total_steps} loss {running:.4f} lr {entry['lr']:.2e} "
                      f"{entry['s_per_step']:.2f} s/step eta {entry['eta_min']:.0f} min", flush=True)
            running = 0.0
            if limit_steps is not None and step >= limit_steps:
                break
        if limit_steps is not None and step >= limit_steps:
            break
    log.close()
    train_seconds = time.perf_counter() - t0

    # ---- save
    clef.language_model.save_pretrained(out_dir / "adapter")
    from safetensors.torch import save_file

    save_file({k: v.detach().cpu().contiguous() for k, v in clef.head.state_dict().items()}, out_dir / "head.safetensors")

    # ---- temperature on the held-out training videos
    calibration: dict[str, Any] = {"temperature": 1.0}
    if val_records:
        logits, target, option_ids = collect_logits(clef, make_loader(val_records, False), device, option_ids_of,
                                                    positive)
        pos = list(option_ids).index(positive)
        temperature = fit_temperature(logits, target)
        calibration = {"temperature": temperature,
                       "val_nll_before": soft_cross_entropy(logits, target).item(),
                       "val_nll_after": soft_cross_entropy(logits / temperature, target).item(),
                       "val_window_auc": window_auc(logits, target, pos), "val_windows": len(val_records)}
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=1))

    run = {"name": data["name"], "machine": os.environ.get("S1_MACHINE") or os.uname().nodename,
           "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
           "started": started, "finished": datetime.now(timezone.utc).isoformat(),
           "train_seconds": train_seconds, "steps": step, "total_steps": total_steps,
           "train_windows": len(train_records), "val_windows": len(val_records), "limit_steps": limit_steps,
           "trainable": {"lora": n_lora, "head": n_head},
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
           "calibration": calibration}
    (out_dir / "run.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(run, indent=1))
    return out_dir
