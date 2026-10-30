#!/usr/bin/env python3
"""Training floor (branch experiment/training-floor, 2026-10-31): the multi-task recipe of the research proposal
(deck p. 61) for d3-nano and Qwen3.5-2B. Arms:
  B  mix:    50 % V1 soft-coverage + 25 % 14-class + 25 % jev-bench replay (ratios from the config)
  C  B + KL: loss += kl_lambda * KL(frozen release ‖ new), reference by disabling the LoRA adapter (and the frozen
             read-out copy for d3) — explicit drift control, two forwards per step
  D  DPO:    -log sigmoid(beta * [(z_chosen - z_rejected) - the same margin under the reference]); rejected = the
             hardest wrong option under the current model

  uv run --frozen --with huggingface_hub python scripts/train-floor.py <config.yaml> [--out DIR] [--limit-steps N]

Config = the stage-1 training config plus:
  tasks: {v1: {n: 10000}, c14: {n: 5000}, replay: {n: 5000, split: train, per_config: 300}}
  anchor: {kl_lambda: 0.5}          # arm C; omit for B
  dpo: {beta: 0.1}                  # arm D; omit otherwise (CE is used)
Batches are homogeneous per task. The 14-class label is the training video's UCF category (normal windows ->
"Normal"): labels at zero cost. Replay items are text-only typed decisions from Praveenrajus/jev-bench train splits;
for the Qwen first-token read-out, items whose options collide on the first token are dropped. Validation, the
temperature and the saved files match the stage-1 trainer (v1 windows only)."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s1stream.assets import resolve  # noqa: E402
from s1stream.config import ConfigError  # noqa: E402
from s1stream.input_forms.frames import read_frames  # noqa: E402
from s1stream.prompt import C14_OPTIONS, PROMPTS  # noqa: E402
from s1stream.train import (D3Trainee, QwenTrainee, fit_temperature, load_train_config, lr_at,  # noqa: E402
                            question_of, soft_cross_entropy)
from s1stream.traindata import Record, read_hivau, sample_records, write_records  # noqa: E402
from s1stream.types import Question  # noqa: E402

C14_BY_FOLDER = {"Normal": "Normal", "Abuse": "Abuse", "Arrest": "Police arrest", "Arson": "Arson",
                 "Assault": "Assault", "Burglary": "Burglary", "Explosion": "Explosion", "Fighting": "Fighting",
                 "RoadAccidents": "Road accident", "Robbery": "Robbery", "Shooting": "Shooting",
                 "Shoplifting": "Shoplifting", "Stealing": "Stealing", "Vandalism": "Vandalism"}
JEV_REPO = "Praveenrajus/jev-bench"


def c14_question() -> Question:
    p = PROMPTS["c14"]
    return Question(state=p["state"], text=p["text"], options=tuple(p["options"]), positive=tuple(p["positive"]))


def replay_items(n_per_config: int, split: str, seed: int) -> list[dict]:
    """Text records from the jev-bench train splits: {id, state, question (decoded dict), options (keys), label}."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(f for f in list_repo_files(JEV_REPO, repo_type="dataset")
                   if f.startswith("data/") and f.endswith(f"/{split}.jsonl"))
    rng = random.Random(seed)
    out = []
    for f in files:
        rows = []
        for line in Path(hf_hub_download(JEV_REPO, f, repo_type="dataset")).read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        rng.shuffle(rows)
        for r in rows[:n_per_config]:
            q = json.loads(r["question"]) if isinstance(r["question"], str) else r["question"]
            state = r["state"]
            if isinstance(state, str) and state[:1] in "{[":
                try:
                    state = json.loads(state)
                except json.JSONDecodeError:
                    pass
            if q["type"] == "noul":
                options, label = ["Yes", "No"], ("Yes" if str(r["label"]) == "1" else "No")
            elif q["type"] == "choice":
                options = list(q["criteria"])
                label = str(r["label"])
                if label not in options:
                    continue
            else:  # score: ordered levels
                options = [str(k) for k in q["criteria"]] if isinstance(q.get("criteria"), dict) else None
                if not options or str(r["label"]) not in options:
                    continue
                label = str(r["label"])
            out.append({"id": r["id"], "state": state, "question": q, "options": options, "label": label})
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Multi-task wrappers: one model, a (rendered prompt, option ids) table per task.


class D3Floor:
    def __init__(self, path: Path, device, lora: dict, grad_checkpoint: bool):
        self.base = D3Trainee(path, device, question_of({"question": {"prompt": "v1", **PROMPTS["v1"]}}),
                              lora, grad_checkpoint)
        self.device = device
        self.code, self.runtime = self.base.code, self.base.runtime
        from s1stream.backbones.d3 import build_questions

        c14 = c14_question()
        self.tables: dict[str, tuple[str, tuple[str, ...]]] = {
            "v1": (self.base.rendered, self.base.option_ids),
            "c14": (self.code.normalize_question(build_questions(c14)["answer"]).rendered, tuple(c14.options)),
        }
        self.states = {"v1": PROMPTS["v1"]["state"], "c14": PROMPTS["c14"]["state"]}
        self.readout0 = self.base.readout.detach().clone()
        self.option_ids = self.base.option_ids  # v1, for validation

    def encode(self, task: str, record, images, payload) -> Any:
        if task == "replay":  # text prompt via the release's own render (d3_format.render: system + user turn)
            normalized = self.code.normalize_question(payload["question"])
            text = self.code.render(self.runtime.processor.tokenizer, self.runtime.prompt, payload["state"],
                                    normalized.rendered, self.runtime.codes)
            return text, [], len(payload["options"])
        text = self.runtime.image_text(self.states[task], self.tables[task][0], len(images))
        return text, images, len(self.tables[task][1])

    def collate(self, encoded: list) -> dict[str, Any]:
        texts = [t for t, _, _ in encoded]
        images = [im for _, imgs, _ in encoded for im in imgs]
        self.runtime.processor.tokenizer.padding_side = "left"
        if images:
            proc = self.runtime.processor(text=texts, images=images, padding=True, return_tensors="pt")
        else:  # replay: text only, tokenizer alone (the processor insists on images)
            proc = self.runtime.processor.tokenizer(texts, padding=True, return_tensors="pt")
        return {**dict(proc), "_n_options": encoded[0][2]}

    def forward(self, batch: dict[str, Any], reference: bool = False) -> torch.Tensor:
        n = batch.pop("_n_options") if "_n_options" in batch else 2
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != "_n_options"}
        ctx = self.base.peft.disable_adapter() if reference else nullcontext()
        with ctx, self.code.sdpa_backends(self.device):
            if "pixel_values" in inputs and self.runtime.encoder is not None:
                embeds, mrope = self.runtime.image_embeddings(inputs)
                hidden = self.runtime.encoder(embeds, inputs["attention_mask"], mrope)[:, -1]
            elif self.runtime.encoder is not None:
                embeds = self.runtime.backbone.get_input_embeddings()(inputs["input_ids"])
                hidden = self.runtime.encoder(embeds, inputs["attention_mask"])[:, -1]
            else:
                hidden = self.runtime.backbone(**inputs, use_cache=False).last_hidden_state[:, -1]
        readout = self.readout0.to(hidden.device) if reference else self.base.readout
        batch["_n_options"] = n
        return (hidden.float() @ readout.T)[:, :n]

    def parameters(self):
        lora, _ = self.base.parameters()
        return lora, [self.base.readout]

    def train(self, mode: bool) -> None:
        self.base.train(mode)

    def save(self, out_dir: Path) -> None:
        self.base.save(out_dir)


class QwenFloor:
    def __init__(self, path: Path, device, lora: dict, grad_checkpoint: bool):
        self.base = QwenTrainee(path, device, question_of({"question": {"prompt": "v1", **PROMPTS["v1"]}}),
                                lora, grad_checkpoint)
        self.device = device
        from s1stream.backbones.qwen_first_token import first_tokens

        self.first_tokens = lambda options: first_tokens(self.base.processor.tokenizer, options)
        c14 = c14_question()
        self.questions = {"v1": question_of({"question": {"prompt": "v1", **PROMPTS["v1"]}}), "c14": c14}
        self.token_tables = {"v1": self.base.token_ids,
                             "c14": torch.tensor(self.first_tokens(self.letter(c14.options)[1]), device=device)}
        self.option_ids = tuple(self.questions["v1"].options)

    @staticmethod
    def letter(options) -> tuple[list[str], list[str]]:
        """(display options, read options): letter-coded when first tokens could collide."""
        display = [f"{chr(65 + i)}. {o}" for i, o in enumerate(options)]
        read = [f"{chr(65 + i)}" for i in range(len(options))]
        return display, display  # read the first token of "A. ..." = "A"

    def accepts(self, options: list[str]) -> bool:
        if len(options) > 26:
            return False
        try:
            self.first_tokens(self.letter(options)[1])
            return True
        except ValueError:
            return False

    def prompt_text(self, state, text, options) -> str:
        q = Question(state=state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
                     text=text, options=tuple(options), positive=options[0])
        messages = [{"role": "user", "content": [{"type": "text", "text": self.base.build_prompt(q)}]}]
        return self.base.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                       enable_thinking=False)

    def encode(self, task: str, record, images, payload) -> Any:
        if task == "replay":
            display, _ = self.letter(payload["options"])
            text = self.prompt_text(payload["state"], payload["question"].get("instructions", ""), display)
            return text, [], self.first_tokens(display)
        q = self.questions[task]
        if task == "c14":
            display, _ = self.letter(list(q.options))
            q = Question(state=q.state, text=q.text, options=tuple(display), positive=display[0])
        messages = [{"role": "user", "content": [{"type": "image"}] * len(images)
                     + [{"type": "text", "text": self.base.build_prompt(q)}]}]
        text = self.base.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                       enable_thinking=False)
        return text, images, self.first_tokens(list(q.options))

    def collate(self, encoded: list) -> dict[str, Any]:
        self.base.processor.tokenizer.padding_side = "left"
        texts = [t for t, _, _ in encoded]
        images = [im for _, imgs, _ in encoded for im in imgs]
        proc = self.base.processor(text=texts, images=images or None, padding=True, return_tensors="pt")
        return {**dict(proc), "_token_ids": torch.tensor(encoded[0][2])}

    def forward(self, batch: dict[str, Any], reference: bool = False) -> torch.Tensor:
        token_ids = batch["_token_ids"].to(self.device)
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != "_token_ids"}
        ctx = self.base.peft.disable_adapter() if reference else nullcontext()
        with ctx:
            logits = self.base.peft(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
        return logits.float()[:, token_ids]

    def parameters(self):
        return self.base.parameters()

    def train(self, mode: bool) -> None:
        self.base.train(mode)

    def save(self, out_dir: Path) -> None:
        self.base.save(out_dir)


FLOORS = {"d3": D3Floor, "qwen_first_token": QwenFloor}


class FloorDataset(torch.utils.data.Dataset):
    """items: (task, record_or_None, payload_or_None, target_index_or_y)."""

    def __init__(self, items, floor):
        self.items, self.floor = items, floor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        task, record, payload, y = self.items[i]
        images = read_frames(Path(record.path), record.frame_indices) if record is not None else []
        return self.floor.encode(task, record, images, payload), task, y


def homogeneous_batches(items, batch: int, seed: int):
    by: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        by.setdefault(it[0], []).append(i)
    rng = random.Random(seed)
    chunks = []
    for task, idx in by.items():
        rng.shuffle(idx)
        chunks += [idx[k:k + batch] for k in range(0, len(idx), batch)]
    rng.shuffle(chunks)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit-steps", type=int, default=None)
    args = ap.parse_args()
    data = load_train_config(args.config)
    out_dir = args.out or Path("results") / os.environ.get("S1_MACHINE", "local") / data["name"] / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dspec, window, lora, optim = data["data"], data["window"], data["lora"], data["optim"]
    tasks = data.get("tasks", {"v1": {"n": int(dspec.get("n_abnormal", 10000)) + int(dspec.get("n_normal", 10000))}})
    anchor = float(data.get("anchor", {}).get("kl_lambda", 0.0))
    dpo_beta = float(data.get("dpo", {}).get("beta", 0.0))
    torch.manual_seed(int(dspec.get("seed", 0)))

    # ---- records
    root = resolve(dspec["asset"])
    items_v = read_hivau(Path(os.path.expandvars(os.path.expanduser(dspec["hivau_json"]))), root,
                         str(dspec.get("videos_dir", "videos")), bool(dspec.get("flat", False)),
                         bool(dspec.get("skip_missing", False)))
    n_v1 = int(tasks.get("v1", {}).get("n", 10000))
    train_records, val_records = sample_records(items_v, float(window["length_s"]), float(window["stride_s"]),
                                                int(window["frames"]), n_v1 // 2, n_v1 - n_v1 // 2,
                                                int(dspec.get("n_val", 2000)), float(dspec.get("holdout", 0.1)),
                                                int(dspec.get("seed", 0)))
    write_records(train_records, out_dir / "records.jsonl")
    write_records(val_records, out_dir / "val.jsonl")

    spec = data["backbone"]
    if spec["name"] not in FLOORS:
        raise ConfigError(f"No floor trainee for backbone {spec['name']!r}")
    path = resolve(spec["asset"]) if "asset" in spec else Path(spec["path"])
    floor = FLOORS[spec["name"]](path, device, lora, bool(optim.get("grad_checkpoint", True)))
    floor.train(True)

    items: list[tuple] = [("v1", r, None, float(r.y)) for r in train_records]
    if "c14" in tasks:
        n = int(tasks["c14"]["n"])
        pool = [r for r in train_records]
        random.Random(int(dspec.get("seed", 0)) + 7).shuffle(pool)
        c14_opts = list(c14_question().options)
        added = 0
        for r in pool:
            if added >= n:
                break
            label = C14_BY_FOLDER.get(r.category, None) if r.y >= 0.5 else "Normal"
            if label is None or label not in c14_opts:
                continue
            items.append(("c14", r, None, c14_opts.index(label)))
            added += 1
        print(f"c14: {added} records", flush=True)
    if "replay" in tasks:
        n = int(tasks["replay"]["n"])
        rep = replay_items(int(tasks["replay"].get("per_config", 400)), str(tasks["replay"].get("split", "train")),
                           int(dspec.get("seed", 0)) + 13)
        kept = 0
        for it in rep:
            if kept >= n:
                break
            if hasattr(floor, "accepts") and not floor.accepts(it["options"]):
                continue
            items.append(("replay", None, it, it["options"].index(it["label"])))
            kept += 1
        print(f"replay: {kept} records", flush=True)
    print(f"total items: {len(items)} ({', '.join(f'{t}:{sum(1 for i in items if i[0] == t)}' for t in dict.fromkeys(i[0] for i in items))})", flush=True)

    batch_size, accumulate = int(optim["batch"]), int(optim.get("accumulate", 1))
    chunks = homogeneous_batches(items, batch_size, int(dspec.get("seed", 0)))
    dataset = FloorDataset(items, floor)

    def collate(parts):
        encs = [p[0] for p in parts]
        return floor.collate(encs), parts[0][1], [p[2] for p in parts]

    loader = torch.utils.data.DataLoader(dataset, batch_sampler=chunks, num_workers=int(optim.get("workers", 6)),
                                         collate_fn=collate, prefetch_factor=4 if int(optim.get("workers", 6)) else None)
    lora_params, head_params = floor.parameters()
    print(f"trainable: LoRA {sum(p.numel() for p in lora_params) / 1e6:.1f} M, head {sum(p.numel() for p in head_params) / 1e6:.1f} M; "
          f"arm: {'C (KL %.2f)' % anchor if anchor else ('D (DPO beta %.2f)' % dpo_beta if dpo_beta else 'B (mix)')}", flush=True)
    epochs = int(optim.get("epochs", 1))
    total_steps = math.ceil(len(chunks) / accumulate) * epochs
    warmup = int(round(total_steps * float(optim.get("warmup", 0.03))))
    groups = [{"params": lora_params, "lr": float(optim["lr"])}]
    if head_params:
        groups.append({"params": head_params, "lr": float(optim.get("head_lr", optim["lr"]))})
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, total_steps, warmup))

    log = (out_dir / "train.log").open("w")
    step, micro, running, t0 = 0, 0, 0.0, time.perf_counter()
    for epoch in range(epochs):
        for batch, task, ys in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = floor.forward(batch)
            n_opt = logits.shape[1]
            if task == "v1":
                y = torch.tensor(ys, device=logits.device, dtype=torch.float32)
                target = torch.stack([y, 1 - y], dim=1)
            else:
                target = torch.zeros(len(ys), n_opt, device=logits.device)
                target[range(len(ys)), torch.tensor(ys, dtype=torch.long)] = 1.0
            need_ref = anchor > 0 or dpo_beta > 0
            if need_ref:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    ref = floor.forward(batch, reference=True)
            if dpo_beta > 0:
                chosen = target.argmax(1)
                masked = logits.detach().clone()
                masked[range(len(ys)), chosen] = -1e9
                rejected = masked.argmax(1)
                margin = logits[range(len(ys)), chosen] - logits[range(len(ys)), rejected]
                ref_margin = ref[range(len(ys)), chosen] - ref[range(len(ys)), rejected]
                loss = -torch.nn.functional.logsigmoid(dpo_beta * (margin - ref_margin)).mean()
            else:
                loss = soft_cross_entropy(logits, target)
                if anchor > 0:
                    ref_p = torch.softmax(ref.float(), dim=-1)
                    loss = loss + anchor * soft_cross_entropy(logits, ref_p) - anchor * (-(ref_p * ref_p.clamp_min(1e-9).log()).sum(1).mean())
            (loss / accumulate).backward()
            running += float(loss.item())
            micro += 1
            if micro % accumulate:
                continue
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total_steps:
                msg = {"step": step, "total": total_steps, "loss": running / (10 * accumulate), "task": task,
                       "s_per_step": (time.perf_counter() - t0) / step}
                print(f"step {step}/{total_steps} loss {msg['loss']:.4f} {msg['s_per_step']:.1f} s/step", flush=True)
                log.write(json.dumps(msg) + "\n"); log.flush(); running = 0.0
            if args.limit_steps and step >= args.limit_steps:
                break
        if args.limit_steps and step >= args.limit_steps:
            break

    # ---- temperature on the held-out v1 windows
    floor.train(False)
    val_items = [("v1", r, None, float(r.y)) for r in val_records]
    vl = torch.utils.data.DataLoader(FloorDataset(val_items, floor), batch_sampler=homogeneous_batches(val_items, batch_size, 1),
                                     num_workers=2, collate_fn=collate)
    zs, ys_all = [], []
    with torch.no_grad():
        for batch, task, ys in vl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                zs.append(floor.forward(batch).float().cpu())
            ys_all += ys
    z = torch.cat(zs) if zs else torch.zeros(0, 2)
    y = torch.tensor([float(v) for v in ys_all])
    t_new = fit_temperature(z, torch.stack([y, 1 - y], dim=1)) if len(z) else 1.0
    (out_dir / "calibration.json").write_text(json.dumps({"temperature": float(t_new), "val_windows": len(z)}, indent=1))
    floor.save(out_dir)
    (out_dir / "run.json").write_text(json.dumps({"name": data["name"], "finished": datetime.now(timezone.utc).isoformat(),
                                                  "steps": step, "arm": "C" if anchor else ("D" if dpo_beta else "B")}, indent=1))
    print(f"done -> {out_dir}  T={float(t_new):.2f}")


if __name__ == "__main__":
    main()
