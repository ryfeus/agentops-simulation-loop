"""Dependency-light contract tests plus optional torch wiring checks in the isolated env."""

from types import SimpleNamespace

import pytest

from experiments.decision_model.model import (
    StrictTokenizer,
    balanced_fixture,
    candidate_predictions,
    changed_groups,
    enable_continuation,
    fit_calibration,
    optimizer_step,
    parameter_groups,
    prediction_delta,
    selection_score,
    temperature_probabilities,
    trainable_snapshot,
    upstream,
)


def prediction(action="refund", label="refund", confidence=0.9, split="calibration"):
    options = ["refund", "escalate", "no_action", "defer"]
    return {
        "action": action,
        "label_action": label,
        "probabilities": {
            key: confidence if key == action else (1 - confidence) / 3 for key in options
        },
        "confidence": confidence,
        "split": split,
        "group_id": "family-a",
        "prohibited_actions": [],
        "error": None,
    }


def test_calibration_cannot_consume_test_or_dev():
    for split in ("train", "dev", "test"):
        with pytest.raises(ValueError, match="disjoint calibration"):
            fit_calibration([prediction(split=split)], {})


def test_calibration_abstains_when_wrong_answer_has_highest_confidence():
    wrong = prediction(action="refund", label="escalate", confidence=0.99)
    wrong["prohibited_actions"] = ["refund"]
    result = fit_calibration([wrong, prediction(confidence=0.6)], {})
    assert result["accepted_count"] == 0
    assert result["threshold"] > 1
    assert result["sample_count"] == 2
    assert result["family_count"] == 1


def test_calibration_accepts_only_correct_high_confidence_rows():
    result = fit_calibration(
        [prediction(confidence=0.99), prediction(action="escalate", confidence=0.5)], {}
    )
    assert result["accepted_count"] == 1
    assert 0 < result["threshold"] <= 1
    assert result["nll_after"] <= result["nll_before"]


def test_selection_prioritizes_prohibited_actions_over_accuracy():
    unsafe = prediction()
    unsafe["prohibited_actions"] = ["refund"]
    safe_wrong = prediction(action="defer")
    assert selection_score([safe_wrong]) > selection_score([unsafe])
    safe_wrong["label_action"] = "defer"
    safe_wrong["error"] = "timeout"
    assert selection_score([safe_wrong])[1] == 0


def test_temperature_preserves_action_and_probability_mass():
    original = prediction()["probabilities"]
    softened = temperature_probabilities(original, 2.0)
    assert sum(softened.values()) == pytest.approx(1)
    assert max(softened, key=softened.get) == "refund"
    assert softened["refund"] < original["refund"]


def test_smoke_fixture_requires_every_class():
    with pytest.raises(ValueError, match="missing class"):
        balanced_fixture([prediction()])


def test_strict_tokenizer_rejects_before_upstream_truncation():
    calls = []

    def tokenizer(texts, **kwargs):
        calls.append(kwargs)
        return {"input_ids": [list(range(len(text))) for text in texts]}

    strict = StrictTokenizer(tokenizer)
    with pytest.raises(ValueError, match="truncation is prohibited"):
        strict(["abcdef"], max_length=3, truncation=True)
    assert len(calls) == 1
    strict(["abc"], max_length=3, truncation=True)
    assert calls[-1]["truncation"] is False


def test_roundtrip_comparison_rejects_errors():
    left = prediction()
    right = prediction()
    assert prediction_delta([left], [right]) == 0
    right["error"] = "broken"
    with pytest.raises(RuntimeError, match="errored"):
        prediction_delta([left], [right])


def test_candidate_metadata_preserves_predictions_and_identifies_in_memory_weights():
    source = {"path": "/source", "files_sha256": {"head": "original"}}
    original = {**prediction(), "checkpoint": source}
    annotated = candidate_predictions([original], source, epoch=2, global_step=58)[0]
    assert annotated["checkpoint"] == {
        "kind": "in_memory_training_candidate",
        "epoch": 2,
        "global_step": 58,
        "source_checkpoint": source,
    }
    assert annotated["source_checkpoint"] == source
    assert original["checkpoint"] == source
    assert {
        key: value
        for key, value in annotated.items()
        if key not in {"checkpoint", "source_checkpoint"}
    } == {key: value for key, value in original.items() if key != "checkpoint"}
    assert selection_score([annotated]) == selection_score([original])


def test_existing_adapter_and_head_train_but_base_stays_exactly_frozen():
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.torso = torch.nn.Module()
            self.torso.base = torch.nn.Linear(2, 2)
            self.torso.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2)})
            self.torso.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2)})
            self.torso.config = SimpleNamespace(use_cache=True)
            self.head = torch.nn.Linear(2, 4)
            self.config = SimpleNamespace(temperature=2.0, temperature_by_kind={"choice": 2.0})

        def forward(self, inputs, labels, temperature):
            hidden = self.torso.base(inputs) + self.torso.lora_B["default"](
                self.torso.lora_A["default"](inputs)
            )
            return {"loss": torch.nn.functional.cross_entropy(self.head(hidden), labels)}

    model = Tiny()
    original_head = model.head
    original_adapter = model.torso.lora_A
    counts = enable_continuation(model)
    assert all(value > 0 for value in counts.values())
    assert model.head is original_head and model.torso.lora_A is original_adapter
    assert model.config.temperature == 1 and model.config.temperature_by_kind == {}
    before = trainable_snapshot(model)
    base_before = model.torso.base.weight.detach().clone()
    optimizer = torch.optim.AdamW(parameter_groups(model, 0.01, 0.01))
    runner = SimpleNamespace(
        torch=torch,
        model=model,
        batch=lambda rows, collator: {
            "inputs": torch.tensor([[1.0, 2.0], [-1.0, 1.0]]),
            "labels": torch.tensor([0, 1]),
        },
    )
    loss, gradients = optimizer_step(runner, optimizer, [[{}]], None)
    assert loss > 0 and all(value > 0 for value in gradients.values())
    assert changed_groups(model, before) == {"head": True, "lora": True}
    assert torch.equal(model.torso.base.weight, base_before)
    assert model.torso.base.weight.grad is None


def test_upstream_option_shuffle_remaps_label_and_pointer_positions():
    torch = pytest.importorskip("torch")
    upstream()
    from training.data import CollatorConfig, Example, SystemOneCollator

    class CharacterTokenizer:
        def __call__(self, texts, **kwargs):
            width = max(len(text) for text in texts)
            ids = [[ord(char) for char in text] + [0] * (width - len(text)) for text in texts]
            offsets = [
                [(i, i + 1) for i in range(len(text))] + [(0, 0)] * (width - len(text))
                for text in texts
            ]
            return {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor([[int(x > 0) for x in row] for row in ids]),
                "offset_mapping": torch.tensor(offsets),
            }

    example = Example(
        kind="choice",
        state="observed disputed invoice",
        instructions="Choose",
        options=[
            ["refund", "refund"],
            ["escalate", "escalate"],
            ["no_action", "no action"],
            ["defer", "defer"],
        ],
        label=1,
    )
    collator = SystemOneCollator(
        CharacterTokenizer(),
        CollatorConfig(head_type="pointer", max_length=1024, seed=42),
        train=True,
    )
    collator._option_order = lambda _: [3, 2, 0, 1]
    batch = collator([example])
    assert batch["labels"].tolist() == [3]
    assert batch["n_slots"].tolist() == [4]
    indices = batch["opt_idx"][0].tolist()
    assert indices == sorted(indices)
    assert all(0 <= index < batch["attention_mask"].sum() for index in indices)
