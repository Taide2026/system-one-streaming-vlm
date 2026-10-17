"""Backbone "gemma_generate" (V2): Gemma 4 E4B that answers by decoding text, unlike the other backbones.

Three set-ups share it:
  prompt: structured, decode: free         the model writes its answer; the head parses it (row "Gemma 4 E4B")
  prompt: structured, decode: constrained  xgrammar allows only an option, then the end of the turn
                                           (row "+ Constrained decoding")
  prompt: question,   decode: free         the question text and "Answer yes or no.", no state or option list
                                           (row "+ Pure prompt")
Decoding is greedy. The backbone returns the answer text; constrained decoding also returns the raw first-step
logits of each option's first token (before the grammar mask), so the head can give p(Yes) instead of 0 / 1.
Constrained decoding uses xgrammar (pinned in pyproject.toml): the grammar root ::= "Yes" | "No" over the model's
tokenizer, applied by xgrammar's transformers logits processor, a new one per generate call as xgrammar requires.
"""
from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any, Callable

import torch

from s1stream.backbones.jev_omni import load_parts
from s1stream.registry import BACKBONES
from s1stream.types import BackboneOutput, ModelInput, Question

PROMPTS = ("structured", "question")
DECODES = ("free", "constrained")
SPECIAL = re.compile(r"<\|?[A-Za-z_]+\|?>")  # <eos>, <turn|>, <|turn>, <|channel>, ...


def build_prompt(question: Question, prompt: str) -> str:
    """structured: f"{state}\\n\\nQUESTION: {text}\\nOPTIONS: {opt1}, {opt2}\\n\\n"
                   "Answer with exactly one of the options and nothing else."
    question:   f"{text} Answer {opt1} or {opt2}." with the options in lower case.
    """
    if prompt == "question":
        return f"{question.text} Answer {' or '.join(option.lower() for option in question.options)}."
    return (f"{question.state}\n\nQUESTION: {question.text}\nOPTIONS: {', '.join(question.options)}\n\n"
            "Answer with exactly one of the options and nothing else.")


def clean(text: str) -> str:
    """Drop a thinking block (everything up to the last "<channel|>") and every special token, then strip."""
    return SPECIAL.sub("", text.split("<channel|>")[-1]).strip()


def options_grammar(options: tuple[str, ...]) -> str:
    """The EBNF grammar of an answer that is exactly one option: root ::= "Yes" | "No"."""
    return "root ::= " + " | ".join(json.dumps(option) for option in options)


def xgrammar_processor_factory(tokenizer: Any, vocab_size: int, eos_ids: list[int]) -> Callable[[tuple[str, ...]], Any]:
    """A function options -> a fresh xgrammar logits processor for one generate call.

    Each options tuple is compiled once against the tokenizer (vocab_size: the logits width) and cached; the end of
    the turn is any of eos_ids.
    """
    import xgrammar as xgr
    from xgrammar.contrib.hf import LogitsProcessor

    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size, stop_token_ids=eos_ids)
    compiler = xgr.GrammarCompiler(info)
    compiled: dict[tuple[str, ...], Any] = {}

    def make(options: tuple[str, ...]) -> Any:
        if options not in compiled:
            compiled[options] = compiler.compile_grammar(options_grammar(options))
        return LogitsProcessor(compiled[options])

    return make


@BACKBONES.register("gemma_generate")
class GemmaGenerate:
    """Config: {name: gemma_generate, asset: gemma-4-e4b, prompt: structured | question, decode: free | constrained,
    max_new_tokens: 16, memory_cap_gb: 40}. load_config adds path."""

    def __init__(self, path: str | Path, prompt: str = "structured", decode: str = "free", max_new_tokens: int = 16,
                 memory_cap_gb: float = 40.0, device: str = "cuda", asset: str | None = None) -> None:
        if device.startswith("cuda"):
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, memory_cap_gb * 2**30 / total_memory))
        model, processor, _ = load_parts(Path(path), device)
        self._init(model, processor, prompt, decode, max_new_tokens, device, None)

    @classmethod
    def from_parts(cls, model: Any, processor: Any, prompt: str = "structured", decode: str = "free",
                   max_new_tokens: int = 16, device: str = "cpu",
                   grammar_processor: Callable[[tuple[str, ...]], Any] | None = None) -> "GemmaGenerate":
        """Build without loading files (used by tests and by __init__). grammar_processor: options -> a logits
        processor for one generate call; None builds xgrammar's from the tokenizer when decode is constrained."""
        instance = cls.__new__(cls)
        instance._init(model, processor, prompt, decode, max_new_tokens, device, grammar_processor)
        return instance

    def _init(self, model: Any, processor: Any, prompt: str, decode: str, max_new_tokens: int, device: str,
              grammar_processor: Callable[[tuple[str, ...]], Any] | None) -> None:
        if prompt not in PROMPTS:
            raise ValueError(f"prompt must be one of {PROMPTS}, got {prompt!r}")
        if decode not in DECODES:
            raise ValueError(f"decode must be one of {DECODES}, got {decode!r}")
        self.model, self.processor, self.device = model, processor, device
        self.prompt, self.decode, self.max_new_tokens = prompt, decode, max_new_tokens
        eos = model.generation_config.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos)
        if decode == "constrained" and grammar_processor is None:
            tokenizer = getattr(processor, "tokenizer", processor)
            grammar_processor = xgrammar_processor_factory(tokenizer, model.config.get_text_config().vocab_size,
                                                           self.eos_ids)
        self.grammar_processor = grammar_processor

    def option_sequences(self, question: Question) -> list[list[int]]:
        """Token ids of each option as the answer's first words, in option order.

        Raise ValueError when two options start with the same token (their first-step logits would be one).
        """
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        sequences = [tokenizer.encode(option, add_special_tokens=False) for option in question.options]
        firsts = [sequence[0] for sequence in sequences]
        if len(set(firsts)) != len(firsts):
            raise ValueError(f"Options share a first token: {dict(zip(question.options, firsts))}")
        return sequences

    def forward(self, inp: ModelInput, question: Question) -> BackboneOutput:
        """1. content = one image entry per frame, then the prompt text; the chat template with thinking off.
        2. Greedy generate (do_sample False), at most max_new_tokens new tokens (constrained: one token per character
           of the longest option plus one, with the xgrammar processor).
        3. Return BackboneOutput(text=clean(new tokens)); constrained also logits = the raw first-step logits of each
           option's first token, in option order.
        """
        content = [{"type": "image", "image": image} for image in inp.images]
        content.append({"type": "text", "text": build_prompt(question, self.prompt)})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
        dtype = next(self.model.parameters()).dtype
        inputs = {key: value.to(self.device, dtype=dtype) if torch.is_floating_point(value)
                  else value.to(self.device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]

        extra: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        sequences = None
        if self.decode == "constrained":
            from transformers import LogitsProcessorList
            sequences = self.option_sequences(question)
            # The grammar may spell an option in several tokens; one per character plus the end always suffices.
            extra = {"max_new_tokens": max(len(option) for option in question.options) + 1,
                     "logits_processor": LogitsProcessorList([self.grammar_processor(tuple(question.options))])}
        autocast = contextlib.nullcontext()
        if self.device.startswith("cuda"):
            autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        with torch.inference_mode(), autocast:
            out = self.model.generate(**inputs, do_sample=False, top_p=None, top_k=None, temperature=None,
                                      return_dict_in_generate=True, output_logits=True, **extra)

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        text = clean(tokenizer.decode(out.sequences[0, prompt_length:].tolist(), skip_special_tokens=False))
        logits = None
        if sequences is not None:
            first = out.logits[0][0].float()
            logits = first[[sequence[0] for sequence in sequences]].detach().cpu()
        return BackboneOutput(text=text, logits=logits)
