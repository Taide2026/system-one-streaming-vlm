"""Unit qwen_first_token: the plain-VLM backbone with a fake model and processor on the CPU. Never loads Qwen."""
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from s1stream.backbones.qwen_first_token import QwenFirstToken, build_prompt, first_tokens
from s1stream.registry import BACKBONES
from s1stream.types import ModelInput, Question

QUESTION = Question(state="Camera, 2 seconds.", text="Abnormal?", options=("Yes", "No"), positive="Yes")
VOCAB = {"Yes": 5, "No": 7, "None": 9, "Nobody": 9}


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [VOCAB[text], 1]


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.messages = self.images = None

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        assert add_generation_prompt and not tokenize and not enable_thinking
        self.messages = messages
        return "<user>...<think>\n\n</think>\n\n"

    def __call__(self, text, images, return_tensors):
        self.images = images
        return {"input_ids": torch.zeros(1, 4, dtype=torch.long)}


class FakeModel:
    def __call__(self, input_ids, logits_to_keep):
        assert logits_to_keep == 1
        return SimpleNamespace(logits=torch.arange(12, dtype=torch.float32).reshape(1, 1, 12))


def test_qwen_first_token_is_registered():
    import s1stream.plugins  # noqa: F401

    assert BACKBONES._classes["qwen_first_token"] is QwenFirstToken


def test_prompt_lists_the_options_in_order():
    assert build_prompt(QUESTION) == "Camera, 2 seconds.\nAbnormal?\nOptions: Yes, No\nAnswer with one option only."


def test_options_must_start_with_distinct_tokens():
    assert first_tokens(FakeTokenizer(), ("Yes", "No")) == [5, 7]
    with pytest.raises(ValueError):
        first_tokens(FakeTokenizer(), ("None", "Nobody"))


def test_forward_reads_each_option_first_token_after_the_frames():
    processor = FakeProcessor()
    imgs = [Image.new("RGB", (32, 24)) for _ in range(8)]
    out = QwenFirstToken.from_parts(FakeModel(), processor).forward(ModelInput(images=imgs), QUESTION)
    assert out.logits.tolist() == [5.0, 7.0]
    content = processor.messages[0]["content"]
    assert content[:8] == [{"type": "image"}] * 8 and content[8]["text"] == build_prompt(QUESTION)
    assert processor.images == imgs
