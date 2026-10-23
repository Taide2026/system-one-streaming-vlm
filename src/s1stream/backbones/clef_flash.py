"""Backbone "clef_flash" (V2): Clef-Flash (Qwen3.5 9B + joint schema head) through the release's own code.

The release ships joint_schema_model.py (revision 8b2e5fd); the backbone imports it from the asset folder and
never downloads. One window is one record:
  {"state": question.state, "images": <frames>,
   "questions": {"answer": {"type": "choice", "instructions": question.text, "criteria": {option: None, ...}}}}
The model scores every option in one forward pass. The release sorts choice options by id; forward returns the
logits in question.options order (head: option_logits).

Streaming (StreamingBackbone, context strategies kv / kv_sinks / kv_compress): the release lays a record out as
  prefix (system prompt + "STATE:\\n") | frames | "\\n" | state | schema | suffix,
so the frames of one window can be prefilled on top of the cached frames of the past windows, and the question
("\\n" + state + schema + suffix) appended last. forward_stream keeps one DynamicCache per video (s1stream.kvcache):
  1. a new video: prefill the prefix;
  2. evict the frames older than policy.keep_windows (never the sinks), prefill the new frames only, compress to
     policy.budget_tokens when set;
  3. snapshot the cache, append the question, run the head on the hidden states of every cached token + the
     question's (the head reads the whole sequence as memory), then restore the snapshot: the cache holds frames only.
Positions are the 3D M-RoPE positions of the whole history (evicted frames keep their place), so with nothing evicted
yet the result is the release's forward on the same frames (checked on the first window of every video).
Only the full-attention layers can forget a token; the Gated DeltaNet layers' recurrent state keeps a trace of every
evicted frame.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from s1stream.context import CachePolicy
from s1stream.kvcache import Segment, StreamState, compress, evict, snapshot
from s1stream.registry import BACKBONES
from s1stream.types import BackboneOutput, ModelInput, Question

QUESTION_ID = "answer"
RELEASE_CODE = "joint_schema_model.py"


def build_record(question: Question, images: list[Any]) -> dict[str, Any]:
    """The Clef record of one window: the frames as images and the question as one choice field."""
    return {
        "state": question.state,
        "images": list(images),
        "questions": {QUESTION_ID: {"type": "choice", "instructions": question.text,
                                    "criteria": {option: None for option in question.options}}},
    }


def load_release_code(path: Path) -> ModuleType:
    """Import path / joint_schema_model.py as a module. Raise FileNotFoundError naming the file when it is missing."""
    file = path / RELEASE_CODE
    if not file.is_file():
        raise FileNotFoundError(file)
    spec = importlib.util.spec_from_file_location("clef_joint_schema_model", file)
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve string annotations through sys.modules, so register before running it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shift_questions(code: ModuleType, questions: tuple, shift: int) -> tuple:
    """The encoded questions with every span moved by shift tokens."""
    return tuple(code.EncodedQuestion(
        question_id=q.question_id, question_type=q.question_type,
        question_span=(q.question_span[0] + shift, q.question_span[1] + shift),
        option_spans=tuple((a + shift, b + shift) for a, b in q.option_spans), option_ids=q.option_ids)
        for q in questions)


@BACKBONES.register("clef_flash")
class ClefFlash:
    """Config: {name: clef_flash, asset: clef-flash, memory_cap_gb: 40}. load_config adds path."""

    def __init__(self, path: str | Path, memory_cap_gb: float = 40.0, device: str = "cuda",
                 asset: str | None = None) -> None:
        """On cuda, cap the GPU memory as jev_omni does; then load_release_model(path, device, local_files_only)."""
        if device.startswith("cuda"):
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, memory_cap_gb * 2**30 / total_memory))
        code = load_release_code(Path(path))
        model, processor = code.load_release_model(Path(path), device=device, local_files_only=True)
        self._init(model, processor, code, device)

    def _init(self, model: Any, processor: Any, code: ModuleType, device: str) -> None:
        self.model, self.processor, self.code, self.device = model, processor, code, device
        self.stream: StreamState | None = None

    @classmethod
    def from_parts(cls, model: Any, processor: Any, code: ModuleType, device: str = "cpu") -> "ClefFlash":
        """Build without loading files (used by tests and by __init__)."""
        instance = cls.__new__(cls)
        instance._init(model, processor, code, device)
        return instance

    def forward(self, inp: ModelInput, question: Question) -> BackboneOutput:
        """encode_record + collate_records + one model call, as the release's `systemone`.

        Return BackboneOutput(logits=<float32 CPU tensor, one logit per option in question.options order>).
        """
        tokenizer = self.processor.tokenizer
        encoded = self.code.encode_record(tokenizer, build_record(question, inp.images), processor=self.processor)
        batch = self.code.collate_records([encoded], tokenizer.pad_token_id, torch.device(self.device))
        with torch.inference_mode():
            logits = self.model(batch)[0][0].float().cpu()
        return self._in_option_order(encoded.questions[0].option_ids, logits, question)

    @staticmethod
    def _in_option_order(option_ids: tuple, logits: torch.Tensor, question: Question) -> BackboneOutput:
        by_option = dict(zip(option_ids, logits.tolist()))
        return BackboneOutput(logits=torch.tensor([by_option[option] for option in question.options]))

    # ---------- streaming: the KV cache across windows ----------

    def start_video(self) -> None:
        """Forget the cache: the next forward_stream starts a new video."""
        self.stream = None

    def forward_stream(self, inp: ModelInput, question: Question, policy: CachePolicy, window: int) -> BackboneOutput:
        """See the module docstring. window is the window's index in its video (0 starts a video)."""
        with torch.inference_mode():
            if self.stream is None or window == 0:
                self.stream = StreamState(cache=self._new_cache())
                prefix = self._prefix_ids()
                self.stream.segments.append(Segment(None, prefix, self._prefill(self.stream, prefix)))
                self.stream.history_ids.extend(prefix)
            state = self.stream
            evict(state, policy, window)

            ids, pixel_values, grid_thw = self._encode_frames(inp.images)
            hidden = self._prefill(state, ids, pixel_values, grid_thw)
            state.segments.append(Segment(window, ids, hidden, len(inp.images)))
            state.history_ids.extend(ids)
            state.history_grid.extend(list(grid_thw))
            if policy.budget_tokens is not None:
                compress(state, policy.budget_tokens)

            question_ids, questions = self._question_segment(question)
            if window == 0:
                self._check_layout(state.cached_ids() + question_ids, question, inp.images)
            saved = snapshot(state.cache)
            question_hidden = self._prefill(state, question_ids)
            state.cache = saved

            all_ids = state.cached_ids() + question_ids
            all_hidden = torch.cat([state.cached_hidden(), question_hidden], dim=0)
            record = self.code.EncodedRecord(input_ids=tuple(all_ids), record_id="stream",
                                             questions=shift_questions(self.code, questions, len(all_ids) - len(question_ids)))
            logits = self._head(all_hidden, all_ids, record)
        return self._in_option_order(record.questions[0].option_ids, logits, question)

    # The model-facing pieces, one method each so tests can stand them in.

    def _qwen(self) -> tuple[Any, Any]:
        """(the Qwen3.5 ForConditionalGeneration, its Qwen3_5Model: visual + language_model + get_rope_index)."""
        language_model = self.model.language_model
        base = language_model.get_base_model() if hasattr(language_model, "get_base_model") else language_model
        return base, base.model

    def _new_cache(self) -> Any:
        from transformers import DynamicCache

        return DynamicCache(config=self._qwen()[1].config)

    def _tokens(self, text: str) -> list[int]:
        return self.processor.tokenizer(text, add_special_tokens=False).input_ids

    def _prefix_ids(self) -> list[int]:
        return self._tokens(f"<|im_start|>system\n{self.code.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")

    def _encode_frames(self, images: list[Any]) -> tuple[list[int], Any, Any]:
        """(token ids of the frames' placeholders, pixel_values, image_grid_thw), as the release encodes media
        (without its trailing newline, which opens the question segment)."""
        encoded = self.processor(text=[self.code.IMAGE_PLACEHOLDER * len(images)], images=images, return_tensors="pt")
        return encoded["input_ids"][0].tolist(), encoded["pixel_values"], encoded["image_grid_thw"]

    def _question_segment(self, question: Question) -> tuple[list[int], tuple]:
        """(ids of "\\n" + state + schema + suffix, the encoded questions with spans relative to this segment)."""
        encoded = self.code.encode_record(self.processor.tokenizer, build_record(question, []))
        prefix_len = len(self._prefix_ids())
        newline = self._tokens("\n")
        ids = newline + list(encoded.input_ids[prefix_len:])
        return ids, shift_questions(self.code, encoded.questions, len(newline) - prefix_len)

    def _check_layout(self, ids: list[int], question: Question, images: list[Any]) -> None:
        """The streamed token layout must be the release's record for the same frames (first window of a video)."""
        encoded = self.code.encode_record(self.processor.tokenizer, build_record(question, images), processor=self.processor)
        if list(encoded.input_ids) != ids:
            raise RuntimeError("Streaming token layout differs from the release's record")

    def _positions(self, ids: list[int], grid_rows: list[Any], start: int) -> Any:
        """The 3D M-RoPE positions of ids[start:], computed over the whole sequence ids (positions run on)."""
        _, qwen = self._qwen()
        all_ids = torch.tensor(ids, device=self.device)[None]
        types = (all_ids == qwen.config.image_token_id).long()
        grid = torch.stack(grid_rows).to(self.device) if grid_rows else None
        positions, _ = qwen.get_rope_index(all_ids, types, image_grid_thw=grid)
        return positions[:, :, start:]

    def _prefill(self, state: StreamState, ids: list[int], pixel_values: Any = None, grid_thw: Any = None) -> Any:
        """Run ids (and their images) through the text model on top of state.cache; return their hidden states
        (tokens, hidden_size). The cache is updated in place; the history is not."""
        _, qwen = self._qwen()
        history = state.history_ids + ids
        grid_rows = state.history_grid + (list(grid_thw) if grid_thw is not None else [])
        positions = self._positions(history, grid_rows, len(state.history_ids))
        media = {}
        if pixel_values is not None:
            media = {"pixel_values": pixel_values.to(self.device), "image_grid_thw": grid_thw.to(self.device)}
        out = qwen(input_ids=torch.tensor(ids, device=self.device)[None], position_ids=positions,
                   past_key_values=state.cache, use_cache=True, return_dict=True, **media)
        return out.last_hidden_state[0]

    def _head(self, hidden: Any, ids: list[int], record: Any) -> torch.Tensor:
        """The joint schema head on the whole sequence; one logit per option, float32 on the CPU."""
        base, _ = self._qwen()
        ids_tensor = torch.tensor(ids, device=self.device)[None]
        logits = self.model.head(hidden[None], ids_tensor, torch.ones_like(ids_tensor), [record],
                                 base.get_output_embeddings().weight)
        return logits[0][0].float().cpu()
