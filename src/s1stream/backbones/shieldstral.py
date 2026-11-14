"""Backbone "shieldstral": Mistral's Shieldstral 1.0 (3.85B; Pixtral vision encoder + Ministral 3B), a yes/no
decision model of another family than d3 (2026-11-13). Second-family check of the upsampling study.

The release answers one yes/no question about a document in one forward pass. Its fixed format (model card):
  system  Judge whether the Document meets the requirements based on the Query and the Instruction provided. ...
  user    <Instruct>: ...\\n\\n<Query>: <one yes/no question>\\n\\n<Document>: <image(s)>
The window's frames are the document; the state line is the instruction and the question text is the query (the
question must be a yes/no question: options ("Yes", "No")). The answer is read as the card does: log-softmax at
the last position, the best of the "yes" token variants against the best of the "no" variants. forward returns
those two values in question.options order (head: option_logits).

Images are encoded at their own size (14-pixel patches, 2x2 merge, no tiling, longest edge capped at 1540), so a
larger input gives more visual tokens, as with d3. The card documents one image per request; several frames are
outside what it documents and are used here as an experiment.
Config: {name: shieldstral, asset: shieldstral-3b, memory_cap_gb: 40}. Local files only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from s1stream.registry import BACKBONES
from s1stream.types import BackboneOutput, ModelInput, Question

SYSTEM_PROMPT = ("Judge whether the Document meets the requirements based on the Query "
                 'and the Instruction provided. Note that the answer can only be "yes" or "no".')
INSTRUCT = "You are reviewing consecutive frames from a video. {state} Judge the frames together as one scene."
VARIANTS = {"yes": ("yes", "yes.", '"yes"', "'yes'"), "no": ("no", "no.", '"no"', "'no'")}


def answer_ids(tokenizer: Any) -> dict[str, list[int]]:
    """Token ids whose text, stripped and lower-cased, is one of the card's yes / no variants."""
    out: dict[str, list[int]] = {"yes": [], "no": []}
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id]).strip().lower()
        for key, variants in VARIANTS.items():
            if text in variants:
                out[key].append(token_id)
    if not out["yes"] or not out["no"]:
        raise ValueError(f"No yes/no tokens found: {out}")
    return out


@BACKBONES.register("shieldstral")
class Shieldstral:
    def __init__(self, path: str | Path, memory_cap_gb: float = 40.0, device: str = "cuda",
                 asset: str | None = None, adapter: str | Path | None = None) -> None:
        from transformers import AutoProcessor, Mistral3ForConditionalGeneration

        if adapter is not None:
            raise ValueError("shieldstral: no fine-tuned adapters yet")
        if device.startswith("cuda"):
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, memory_cap_gb * 2**30 / total_memory))
        self.model = Mistral3ForConditionalGeneration.from_pretrained(Path(path), local_files_only=True,
                                                                      dtype=torch.bfloat16).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(Path(path), local_files_only=True)
        self.device = device
        self.ids = answer_ids(self.processor.tokenizer)

    def prepare(self, images: list, question: Question) -> dict:
        """Processor inputs of one window, on the device."""
        if tuple(o.lower() for o in question.options) != ("yes", "no"):
            raise ValueError(f"shieldstral needs a yes/no question, got {question.options}")
        prefix = f"<Instruct>: {INSTRUCT.format(state=question.state)}\n\n<Query>: {question.text}\n\n<Document>: "
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": prefix}] + [{"type": "image"}] * len(images)}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[rendered], images=list(images), return_tensors="pt")
        return {key: (value.to(self.device) if hasattr(value, "to") else value) for key, value in inputs.items()}

    def read(self, logits: torch.Tensor, question: Question) -> torch.Tensor:
        logp = torch.log_softmax(logits.float(), dim=-1)
        z = {key: logp[ids].max() for key, ids in self.ids.items()}
        return torch.stack([z[o.lower()] for o in question.options]).detach().cpu()

    def forward(self, inp: ModelInput, question: Question) -> BackboneOutput:
        inputs = self.prepare(list(inp.images), question)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1]
        return BackboneOutput(logits=self.read(logits, question))
