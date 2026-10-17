"""Unit gemma_generate: the decoding backbone and the generated_answer head, with tiny fakes on the CPU.
Never loads Gemma."""
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from s1stream.heads.generated_answer import GeneratedAnswer, parse
from s1stream.types import BackboneOutput, ModelInput, Question

QUESTION = Question(state="Camera, 2 seconds.", text="Abnormal?", options=("Yes", "No"), positive="Yes")
EOS = 1
VOCAB = {1: "<turn|>", 5: "Maybe", 10: "Yes", 11: "No", 12: ".", 20: "<|channel>", 21: "thought", 22: "<channel|>"}
PROMPT_IDS = [2, 3, 4]


def images(n: int = 8) -> list:
    return [Image.new("RGB", (32, 24), (i * 20, 0, 0)) for i in range(n)]


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        ids = {"Yes": [10], "No": [11], "Yes please": [10, 5], "Yesterday": [10, 12]}
        return list(ids[text])

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(VOCAB[i] for i in ids)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return {"input_ids": torch.tensor([PROMPT_IDS]), "pixel_values": torch.zeros(1, 3, 4, 4)}


class FakeModel(torch.nn.Module):
    """Greedy decoding over scripted raw logits, one row per step; applies a logits processor like generate does."""

    def __init__(self, steps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.steps = steps
        self.generation_config = SimpleNamespace(eos_token_id=[EOS, 106])
        self.kwargs = None

    def generate(self, input_ids, pixel_values, max_new_tokens, logits_processor=None, **kwargs):
        self.kwargs = dict(kwargs, max_new_tokens=max_new_tokens, logits_processor=logits_processor)
        sequence, raw = input_ids, []
        for step in range(max_new_tokens):
            scores = torch.full((1, 128), -10.0)
            for token, value in self.steps[min(step, len(self.steps) - 1)].items():
                scores[0, token] = value
            raw.append(scores.clone())
            if logits_processor is not None:
                for processor in logits_processor:
                    scores = processor(sequence, scores)
            token = int(scores.argmax(-1))
            sequence = torch.cat([sequence, torch.tensor([[token]])], dim=1)
            if token == EOS:
                break
        return SimpleNamespace(sequences=sequence, logits=tuple(raw))


class FakeGrammar:
    """Stands in for xgrammar's processor on the fake vocabulary: Yes (10) or No (11), then the end of the turn."""

    def __init__(self, options):
        self.options = options
        self.steps = 0

    def __call__(self, input_ids, scores):
        allowed = [10, 11] if self.steps == 0 else [EOS, 106]
        self.steps += 1
        mask = torch.full_like(scores, float("-inf"))
        mask[:, [a for a in allowed if a < scores.shape[1]]] = 0
        return scores + mask


def backbone(steps, **kwargs):
    model, processor = FakeModel(steps), FakeProcessor()
    if kwargs.get("decode") == "constrained":
        kwargs.setdefault("grammar_processor", FakeGrammar)
    return GemmaGenerate.from_parts(model, processor, **kwargs), model, processor


# ---------- generated_answer ----------

@pytest.mark.parametrize("text, answer", [
    ("Yes", "Yes"), ("no.", "No"), ("**No**, nothing unusual.", "No"), ("Yes, a man is hitting a woman.", "Yes"),
    ("The answer is yes.", "Yes"), ("Nothing abnormal", None), ("yes or no", "Yes"), ("It could be yes or no", None),
    ("", None),
])
def test_parse(text, answer):
    assert parse(text, QUESTION.options) == answer


def test_head_one_hot_for_a_parsed_answer_and_zeros_when_invalid():
    head = GeneratedAnswer()
    assert head.probabilities(BackboneOutput(text="No."), QUESTION) == {"Yes": 0.0, "No": 1.0}
    assert head.probabilities(BackboneOutput(text="I cannot tell."), QUESTION) == {"Yes": 0.0, "No": 0.0}


def test_head_softmax_over_option_logits():
    probs = GeneratedAnswer().probabilities(BackboneOutput(text="Yes", logits=torch.tensor([2.0, 0.0])), QUESTION)
    assert probs["Yes"] == pytest.approx(torch.tensor([2.0, 0.0]).softmax(-1)[0].item())


def test_head_rejects_an_empty_output_or_wrong_logit_count():
    with pytest.raises(ValueError):
        GeneratedAnswer().probabilities(BackboneOutput(), QUESTION)
    with pytest.raises(ValueError):
        GeneratedAnswer().probabilities(BackboneOutput(logits=torch.zeros(3)), QUESTION)
