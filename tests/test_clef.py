"""Unit clef: the clef_flash backbone (one window, and streaming across windows) and the option_logits head, with tiny
fakes on the CPU. Never loads Clef-Flash."""
import copy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from s1stream.backbones.clef_flash import ClefFlash, build_record, load_release_code
from s1stream.context import CachePolicy
from s1stream.heads.option_logits import OptionLogits
from s1stream.types import BackboneOutput, ModelInput, Question

QUESTION = Question(state="Camera, 2 seconds.", text="Abnormal?", options=("Yes", "No"), positive="Yes")


def images(n: int = 8) -> list:
    return [Image.new("RGB", (32, 24), (i * 20, 0, 0)) for i in range(n)]


# ---------- option_logits ----------

def test_option_logits_softmax_in_option_order():
    probs = OptionLogits().probabilities(BackboneOutput(logits=torch.tensor([2.0, 0.0])), QUESTION)
    assert list(probs) == ["Yes", "No"]
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["Yes"] == pytest.approx(torch.tensor([2.0, 0.0]).softmax(-1)[0].item())


def test_option_logits_rejects_missing_or_wrong_count():
    with pytest.raises(ValueError):
        OptionLogits().probabilities(BackboneOutput(), QUESTION)
    with pytest.raises(ValueError):
        OptionLogits().probabilities(BackboneOutput(logits=torch.zeros(3)), QUESTION)


# ---------- clef_flash, one window ----------

@dataclass
class FakeEncodedQuestion:
    option_ids: tuple


class FakeClefCode:
    """Stands in for joint_schema_model.py: options sorted by id, as the release does for choice fields."""

    def __init__(self):
        self.record = None

    def encode_record(self, tokenizer, record, processor=None):
        self.record = record
        options = sorted(record["questions"]["answer"]["criteria"])
        return SimpleNamespace(questions=(FakeEncodedQuestion(tuple(options)),))

    def collate_records(self, records, pad_token_id, device):
        return {"records": records}


class FakeClefModel:
    def __init__(self, logits):
        self.logits = logits
        self.grad_enabled = None

    def __call__(self, batch):
        self.grad_enabled = torch.is_grad_enabled()
        return [[self.logits]]


def test_clef_record_has_frames_state_and_one_choice_question():
    imgs = images()
    record = build_record(QUESTION, imgs)
    assert record["state"] == QUESTION.state
    assert record["images"] == imgs
    assert record["questions"] == {"answer": {"type": "choice", "instructions": "Abnormal?",
                                              "criteria": {"Yes": None, "No": None}}}


def test_clef_forward_returns_logits_in_question_order_without_grad():
    code = FakeClefCode()
    model = FakeClefModel(torch.tensor([0.5, 2.0]))  # the release's order: No, Yes
    backbone = ClefFlash.from_parts(model, SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)), code)
    out = backbone.forward(ModelInput(images=images()), QUESTION)
    assert out.logits.tolist() == [2.0, 0.5]  # Yes, No
    assert model.grad_enabled is False
    assert code.record["images"] == backbone.code.record["images"]


def test_clef_missing_release_code_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_release_code(tmp_path)


# ---------- clef_flash, streaming ----------
# A fake release: tokens are small ints. prefix = [1, 2]; an image = [VS, PAD, PAD, VE]; the newline = [9];
# state = one token per character code offset; schema = [20, 21 (question), 22, 30 (option No), 31 (option Yes), 23];
# suffix = [40]. The fake text model returns hidden = token id as a one-element vector, so the head can check what
# it was given.

VS, PAD, VE, NL = 100, 101, 102, 9
PREFIX = [1, 2]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple
    option_spans: tuple
    option_ids: tuple


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple
    questions: tuple
    record_id: str
    media: object = None


class StreamCode:
    SYSTEM_PROMPT = "sys"
    IMAGE_PLACEHOLDER = "<img>"
    EncodedQuestion = EncodedQuestion
    EncodedRecord = EncodedRecord

    @staticmethod
    def state_ids(state):
        return [200 + i for i in range(len(state) % 3 + 1)]

    def encode_record(self, tokenizer, record, processor=None):
        media = [VS, PAD, PAD, VE] * len(record["images"]) + ([NL] if record["images"] else [])
        state = self.state_ids(record["state"])
        base = len(PREFIX) + len(media) + len(state)
        schema = [20, 21, 22, 30, 31, 23]
        ids = PREFIX + media + state + schema + [40]
        question = EncodedQuestion("answer", 1, (base + 1, base + 2), ((base + 3, base + 4), (base + 4, base + 5)),
                                   ("No", "Yes"))
        return EncodedRecord(tuple(ids), (question,), "r")


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        if text.endswith("STATE:\n"):
            return SimpleNamespace(input_ids=PREFIX)
        if text == "\n":
            return SimpleNamespace(input_ids=[NL])
        raise AssertionError(text)


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __call__(self, text, images, return_tensors):
        n = len(images)
        return {"input_ids": torch.tensor([[VS, PAD, PAD, VE] * n]), "pixel_values": torch.zeros(n, 3),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * n)}


class FakeCache:
    """Tokens as a list; the attention layer's keys are their ids."""

    def __init__(self):
        self.layers = [SimpleNamespace(keys=torch.zeros(1, 1, 0, 1), values=torch.zeros(1, 1, 0, 1))]

    def append(self, ids):
        new = torch.tensor(ids, dtype=torch.float32).view(1, 1, -1, 1)
        self.layers[0].keys = torch.cat([self.layers[0].keys, new], dim=-2)
        self.layers[0].values = self.layers[0].keys.clone()

    def ids(self):
        return [int(v) for v in self.layers[0].keys.flatten().tolist()]


class StreamingFake(ClefFlash):
    """ClefFlash with the model-facing pieces replaced; records every head call."""

    def __init__(self):
        self._init(model=None, processor=FakeProcessor(), code=StreamCode(), device="cpu")
        self.head_calls = []
        self.position_starts = []

    def _new_cache(self):
        return FakeCache()

    def _prefill(self, state, ids, pixel_values=None, grid_thw=None):
        self.position_starts.append(len(state.history_ids))
        state.cache.append(ids)
        return torch.tensor(ids, dtype=torch.float32).view(-1, 1)

    def _head(self, hidden, ids, record):
        self.head_calls.append((hidden.flatten().tolist(), list(ids), record))
        return torch.tensor([0.5, 2.0])  # No, Yes


def test_stream_first_window_has_the_release_layout_and_reads_yes_no_in_option_order():
    backbone = StreamingFake()
    out = backbone.forward_stream(ModelInput(images=images(2)), QUESTION, CachePolicy(keep_windows=1), window=0)
    assert out.logits.tolist() == [2.0, 0.5]
    hidden, ids, record = backbone.head_calls[0]
    expected = StreamCode().encode_record(None, build_record(QUESTION, images(2)))
    assert tuple(ids) == expected.input_ids  # prefix | frames | newline | state | schema | suffix
    assert record.questions == expected.questions
    assert hidden == [float(i) for i in ids]  # the head sees every cached token and the question's
    # the question's tokens were dropped again: the cache holds prefix + frames only
    assert backbone.stream.cache.ids() == PREFIX + [VS, PAD, PAD, VE] * 2


def test_stream_evicts_old_windows_but_positions_run_on():
    backbone = StreamingFake()
    frames = [VS, PAD, PAD, VE] * 2
    for window in range(3):
        backbone.forward_stream(ModelInput(images=images(2)), QUESTION, CachePolicy(keep_windows=1), window=window)
    # window 2: prefix + frames of windows 1 and 2 in the cache; the head saw the same
    assert backbone.stream.cache.ids() == PREFIX + frames * 2
    assert [s.window for s in backbone.stream.segments] == [None, 1, 2]
    _, ids, record = backbone.head_calls[-1]
    assert ids[:len(PREFIX) + 2 * len(frames)] == PREFIX + frames * 2
    assert record.questions[0].option_spans[1] == (len(ids) - 3, len(ids) - 2)  # the Yes option, before END FIELD + suffix
    # positions: the evicted window 0 keeps its place, so window 2's frames start after prefix + 2 windows
    frame_prefills = backbone.position_starts[1::2][:3]  # prefix, (frames, question) x 3 -> frames at 1, 3, 5
    assert frame_prefills == [len(PREFIX), len(PREFIX) + len(frames), len(PREFIX) + 2 * len(frames)]


def test_stream_window_zero_starts_a_new_video():
    backbone = StreamingFake()
    backbone.forward_stream(ModelInput(images=images(1)), QUESTION, CachePolicy(keep_windows=3), window=0)
    backbone.forward_stream(ModelInput(images=images(1)), QUESTION, CachePolicy(keep_windows=3), window=1)
    backbone.forward_stream(ModelInput(images=images(1)), QUESTION, CachePolicy(keep_windows=3), window=0)
    assert [s.window for s in backbone.stream.segments] == [None, 0]


def test_stream_layout_check_fails_loudly():
    class BadCode(StreamCode):
        def encode_record(self, tokenizer, record, processor=None):
            encoded = super().encode_record(tokenizer, record, processor)
            if record["images"]:
                return EncodedRecord(encoded.input_ids + (7,), encoded.questions, "r")  # the release did something else
            return encoded

    backbone = StreamingFake()
    backbone.code = BadCode()
    with pytest.raises(RuntimeError):
        backbone.forward_stream(ModelInput(images=images(1)), QUESTION, CachePolicy(keep_windows=1), window=0)
