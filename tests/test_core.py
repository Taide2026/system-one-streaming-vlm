"""Unit core: registry and config."""
from pathlib import Path

import pytest
import yaml

from s1stream.config import ConfigError, ExperimentConfig, load_config
from s1stream.datasets.xd_violence import LABELS
from s1stream.prompt import PROMPTS, V2_CLASSES, V2_XD_CLASSES
from s1stream.registry import Registry, RegistryError
from s1stream.types import Question

EXAMPLE = {
    "name": "demo",
    "dataset": {"name": "ucf_crime", "n_anomaly": 1, "n_normal": 1, "max_duration_s": 60, "seed": 1},
    "window": {"length_s": 2.0, "stride_s": 2.0, "frames": 8},
    "question": {"state": "Camera.", "text": "Abnormal?", "options": ["Yes", "No"], "positive": "Yes"},
    "input_form": {"name": "frames"},
    "backbone": {"name": "fake"},
    "answer_head": {"name": "option_prob"},
    "decision_rules": [{"name": "threshold", "start": 0.0, "stop": 1.0, "step": 0.5}],
}


def write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_register_and_build_with_params():
    reg = Registry("demo")

    @reg.register("x")
    class X:
        def __init__(self, a):
            self.a = a

    built = reg.build({"name": "x", "a": 1})
    assert isinstance(built, X)
    assert built.a == 1


def test_build_unknown_name_lists_known_names():
    reg = Registry("demo")
    reg.register("alpha")(type("A", (), {}))
    reg.register("beta")(type("B", (), {}))
    with pytest.raises(RegistryError) as err:
        reg.build({"name": "gamma"})
    assert "alpha" in str(err.value)
    assert "beta" in str(err.value)


def test_register_same_name_twice_raises():
    reg = Registry("demo")
    reg.register("x")(type("A", (), {}))
    with pytest.raises(RegistryError):
        reg.register("x")(type("B", (), {}))


def test_load_config_reads_all_sections(tmp_path):
    cfg = load_config(write(tmp_path, EXAMPLE), resolver=lambda name: tmp_path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "demo"
    assert cfg.dataset["n_anomaly"] == 1
    assert (cfg.window.length_s, cfg.window.stride_s, cfg.window.frames) == (2.0, 2.0, 8)
    assert cfg.question.options == ("Yes", "No")
    assert cfg.question.positive == "Yes"
    assert cfg.input_form == {"name": "frames"}
    assert cfg.backbone == {"name": "fake"}
    assert cfg.answer_head == {"name": "option_prob"}
    assert cfg.decision_rules[0]["step"] == 0.5
    assert cfg.raw == EXAMPLE


def test_load_config_missing_section_names_it(tmp_path):
    data = {k: v for k, v in EXAMPLE.items() if k != "backbone"}
    with pytest.raises(ConfigError) as err:
        load_config(write(tmp_path, data), resolver=lambda name: tmp_path)
    assert "backbone" in str(err.value)


def test_load_config_positive_not_in_options(tmp_path):
    data = dict(EXAMPLE, question=dict(EXAMPLE["question"], positive="Maybe"))
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, data), resolver=lambda name: tmp_path)


CLASSES = {"state": "Camera.", "text": "What is happening?", "options": ["Normal", "Fighting", "Arson"],
           "positive": ["Fighting", "Arson"]}


def test_load_config_positive_list_becomes_a_tuple(tmp_path):
    cfg = load_config(write(tmp_path, dict(EXAMPLE, question=CLASSES)), resolver=lambda name: tmp_path)
    assert cfg.question.positive == ("Fighting", "Arson")


@pytest.mark.parametrize("positive", [[], ["Fighting", "Maybe"], ["Arson", "Arson"]])
def test_load_config_positive_list_must_be_distinct_options(tmp_path, positive):
    data = dict(EXAMPLE, question=dict(CLASSES, positive=positive))
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, data), resolver=lambda name: tmp_path)


def test_question_prompt_fills_the_question_and_raw(tmp_path):
    cfg = load_config(write(tmp_path, dict(EXAMPLE, question={"prompt": "v2"})), resolver=lambda name: tmp_path)
    assert cfg.question.options == tuple(V2_CLASSES) + ("None",)
    assert cfg.question.score({option: 1 / 14 for option in cfg.question.options}) == pytest.approx(13 / 14)
    assert cfg.raw["question"] == {"prompt": "v2", **PROMPTS["v2"]}


def test_xd_prompts_name_the_xd_classes_with_none_last():
    assert set(V2_XD_CLASSES.values()) == set(LABELS.values())
    assert PROMPTS["v2_xd"]["options"] == [*V2_XD_CLASSES, "None"]
    assert PROMPTS["v1_xd"]["options"] == ["Yes", "No"]


def test_question_prompt_keys_in_the_config_win(tmp_path):
    data = dict(EXAMPLE, question={"prompt": "v1", "state": "Camera."})
    cfg = load_config(write(tmp_path, data), resolver=lambda name: tmp_path)
    assert (cfg.question.state, cfg.question.options, cfg.question.positive) == ("Camera.", ("Yes", "No"), "Yes")
    assert cfg.question.text == PROMPTS["v1"]["text"]


def test_unknown_question_prompt_raises(tmp_path):
    with pytest.raises(ConfigError, match="question.prompt"):
        load_config(write(tmp_path, dict(EXAMPLE, question={"prompt": "v9"})), resolver=lambda name: tmp_path)


def test_question_score_is_the_positive_probability_or_their_sum():
    probabilities = {"Normal": 0.5, "Fighting": 0.3, "Arson": 0.2}
    one = Question(state="s", text="q", options=("Normal", "Fighting", "Arson"), positive="Fighting")
    many = Question(state="s", text="q", options=("Normal", "Fighting", "Arson"), positive=("Fighting", "Arson"))
    assert one.score(probabilities) == pytest.approx(0.3)
    assert many.score(probabilities) == pytest.approx(0.5)
