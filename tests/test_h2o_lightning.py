"""Unit h2o_lightning: the H2O-Lightning-4B backbone with a fake model and processor on the CPU. Never loads it."""
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from s1stream.backbones.h2o_lightning import H2OLightning, build_prompt, label_ids, spread
from s1stream.registry import BACKBONES
from s1stream.types import ModelInput, Question

QUESTION = Question(state="Camera, 2 seconds.", text="Abnormal?", options=("Yes", "No"), positive="Yes")
SERVE = {"temperature": 0.8, "labels": ["A", "B", "C"], "image": {"max_images": 4, "temperature": 0.5},
         "prompt": {"system": "Decide.", "prefill": "Answer:"}}


class FakeTokenizer:
    """One token per character, " A" / " B" one token each."""
    def encode(self, text, add_special_tokens=False):
        for label, token in ((" A", 100), (" B", 101), (" C", 102)):
            if text.endswith(label):
                return self.encode(text[:-2]) + [token]
        return [ord(c) for c in text]


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.text = self.images = None

    def __call__(self, text, images, return_tensors):
        self.text, self.images = text[0], images
        return {"input_ids": torch.zeros(1, 4, dtype=torch.long)}


class FakeModel:
    def __init__(self):
        weight = torch.zeros(200, 2)
        weight[100], weight[101] = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
        self.lm_head = SimpleNamespace(weight=weight)
        self.model = lambda **inputs: SimpleNamespace(last_hidden_state=torch.tensor([[[0.0, 0.0], [2.0, 1.0]]]))


def test_h2o_lightning_is_registered():
    import s1stream.plugins  # noqa: F401

    assert BACKBONES._classes["h2o_lightning"] is H2OLightning


def test_spread_keeps_the_first_and_last_frame():
    assert spread(8, 4) == [0, 2, 5, 7]
    assert spread(3, 4) == [0, 1, 2]


def test_prompt_is_the_shim_image_prompt():
    prompt = build_prompt(SERVE, QUESTION, 2)
    assert prompt == ("<|im_start|>system\nDecide.<|im_end|>\n<|im_start|>user\n"
                      + "<|vision_start|><|image_pad|><|vision_end|>" * 2
                      + "record: Camera, 2 seconds.\nquestion: Abnormal?\noptions:\nA) Yes: Yes\nB) No: No<|im_end|>\n"
                      "<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:")


def test_labels_must_be_one_token_at_the_answer_slot():
    assert label_ids(FakeTokenizer(), "Answer:", ["A", "B"]) == [100, 101]
    with pytest.raises(ValueError):
        label_ids(FakeTokenizer(), "Answer:", ["Z"])


def test_forward_reads_four_frames_and_the_label_logits_over_the_temperature():
    processor = FakeProcessor()
    imgs = [Image.new("RGB", (32, 24), (i, 0, 0)) for i in range(8)]
    out = H2OLightning.from_parts(FakeModel(), processor, SERVE).forward(ModelInput(images=imgs), QUESTION)
    assert processor.images == [imgs[0], imgs[2], imgs[5], imgs[7]]
    assert processor.text == build_prompt(SERVE, QUESTION, 4)
    assert out.logits.tolist() == pytest.approx([4.0, 2.0])
