"""Unit gemma_generate: the decoding backbone and the generated_answer head, with tiny fakes on the CPU.
Never loads Gemma."""
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from s1stream.backbones.gemma_generate import (GemmaGenerate, build_prompt, clean, options_grammar,
                                               xgrammar_processor_factory)
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


# ---------- prompt and text ----------

def test_structured_prompt_has_state_question_options_and_rule():
    assert build_prompt(QUESTION, "structured") == (
        "Camera, 2 seconds.\n\nQUESTION: Abnormal?\nOPTIONS: Yes, No\n\n"
        "Answer with exactly one of the options and nothing else.")


def test_pure_prompt_is_the_question_and_answer_yes_or_no():
    assert build_prompt(QUESTION, "question") == "Abnormal? Answer yes or no."


def test_clean_drops_thinking_block_and_special_tokens():
    assert clean("<|channel>thought\nhmm<channel|>Yes.<turn|>") == "Yes."
    assert clean(" No <eos>") == "No"


def test_bad_prompt_or_decode_is_rejected():
    with pytest.raises(ValueError):
        backbone([{}], prompt="json")
    with pytest.raises(ValueError):
        backbone([{}], decode="beam")


# ---------- free decoding ----------

def test_free_decoding_returns_the_cleaned_text_and_no_logits():
    gemma, model, processor = backbone([{10: 5.0}, {12: 5.0}, {EOS: 5.0}])
    out = gemma.forward(ModelInput(images=images(3)), QUESTION)
    assert out.text == "Yes ."
    assert out.logits is None
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["max_new_tokens"] == 16
    assert model.kwargs["logits_processor"] is None


def test_frames_come_first_then_the_prompt_with_thinking_off():
    gemma, _, processor = backbone([{EOS: 5.0}], prompt="question")
    imgs = images(16)
    gemma.forward(ModelInput(images=imgs), QUESTION)
    content = processor.messages[0]["content"]
    assert [part["image"] for part in content[:-1]] == imgs
    assert content[-1] == {"type": "text", "text": "Abnormal? Answer yes or no."}
    assert processor.kwargs["enable_thinking"] is False
    assert processor.kwargs["add_generation_prompt"] is True


def test_free_decoding_stops_at_max_new_tokens():
    gemma, _, _ = backbone([{5: 5.0}], max_new_tokens=3)
    assert gemma.forward(ModelInput(images=images(1)), QUESTION).text == "Maybe Maybe Maybe"


# ---------- constrained decoding ----------

def test_options_grammar_is_one_option():
    assert options_grammar(("Yes", "No")) == 'root ::= "Yes" | "No"'


def test_xgrammar_allows_only_an_option_then_the_end():
    """Real xgrammar on a tiny word-level tokenizer: the first step allows only tokens that start an option; after
    "Yes" only the end of the turn."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    vocab = {"<eos>": 0, "Yes": 1, "No": 2, "Maybe": 3, "Y": 4, "es": 5, "N": 6, "o": 7}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<eos>")),
                                        eos_token="<eos>")
    make = xgrammar_processor_factory(tokenizer, vocab_size=len(vocab), eos_ids=[0])
    rule = make(("Yes", "No"))
    prompt = torch.tensor([[3]])
    first = rule(prompt, torch.zeros(1, len(vocab)))
    assert set(torch.isfinite(first[0]).nonzero().flatten().tolist()) == {1, 2, 4, 6}
    after_yes = rule(torch.tensor([[3, 1]]), torch.zeros(1, len(vocab)))
    assert set(torch.isfinite(after_yes[0]).nonzero().flatten().tolist()) == {0}


def test_constrained_decoding_picks_an_option_even_when_another_token_wins():
    # Raw logits prefer "Maybe" (5); the rule leaves only Yes (10) / No (11), and No is higher.
    gemma, model, _ = backbone([{5: 9.0, 10: 1.0, 11: 2.0}, {5: 9.0}], decode="constrained")
    out = gemma.forward(ModelInput(images=images(2)), QUESTION)
    assert out.text == "No"
    assert model.kwargs["max_new_tokens"] == 4  # one per character of the longest option ("Yes") plus the end
    assert torch.equal(out.logits, torch.tensor([1.0, 2.0]))  # raw first-step logits, option order


def test_constrained_decoding_rejects_options_sharing_a_first_token():
    gemma, _, _ = backbone([{10: 1.0}], decode="constrained")
    question = Question(state="s", text="q", options=("Yes", "Yesterday"), positive="Yes")
    with pytest.raises(ValueError):
        gemma.forward(ModelInput(images=images(1)), question)


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
