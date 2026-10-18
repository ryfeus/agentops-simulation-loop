"""Semantic root splits precede snapshots; teacher evidence never enters model input."""

import json
from collections import Counter, defaultdict

import pytest

from experiments.decision_model import data, history, next_data


def test_frozen_design_roots_and_historical_official_train_only():
    planned = next_data.planned_tasks()
    counts = Counter(split for _, split, *_ in planned)
    assert counts == {"train": 175, "dev": 12, "calibration": 24, "test": 12}
    roots = defaultdict(set)
    structures = defaultdict(set)
    for task, split, root, _, source in planned:
        roots[root].add(split)
        structures[next_data.structural_signature(task)].add(split)
        if source != "fresh-authored-composition":
            assert split == "train"
        assert task.id not in next_data.EXCLUDED_HISTORICAL
    assert all(len(splits) == 1 for splits in roots.values())
    assert all(len(splits) == 1 for splits in structures.values())
    for split, expected in [("dev", 4), ("calibration", 4), ("test", 6)]:
        assert sum(splits == {split} for splits in roots.values()) == expected
    assert (
        len({next_data.structural_signature(t) for t, s, _, _, _ in planned if s != "train"}) == 14
    )


def test_new_variants_share_composition_structure_not_new_independent_roots():
    for prototype in next_data.NEW_PROTOTYPES:
        tasks = [
            next_data.new_scenario(prototype, i)[0]
            for i in range(next_data.VARIANTS[prototype["split"]])
        ]
        assert len({next_data.structural_signature(t) for t in tasks}) == 1
        assert len({t.id for t in tasks}) == len(tasks)
        assert len({next_data.normalized_instruction(t.instruction) for t in tasks}) == 1
        assert len({t.benchmark.generation.prototype_sha256 for t in tasks}) == 1


def observed(status="paid", records=()):
    return history.make_state(
        "Refund inv-1.",
        "inv-1",
        {
            "ok": True,
            "invoice": {"id": "inv-1", "status": status},
            "escalation_history": {
                "invoice_id": "inv-1",
                "complete": True,
                "records": list(records),
            },
        },
        [],
    )


def metadata(operation="refund_invoice"):
    return {
        "required_inspections": ["inv-1"],
        "expected_operations": {"inv-1": operation},
        "defer_targets": [],
    }


def test_labels_are_not_request_parser_self_labels_or_auth_feature_labels():
    state = observed()
    state["authorization"] = {"status": "denied", "allowed_actions": []}
    assert next_data.label_state(state, metadata()) == "refund"
    state["authorization"] = {"status": "authorized", "allowed_actions": ["escalate"]}
    assert next_data.label_state(state, metadata()) == "refund"
    assert next_data.label_state(state, {**metadata(), "defer_targets": ["inv-1"]}) == "defer"


def test_rollout_labels_require_actual_observations_and_successful_completion():
    state = observed()
    state["prior_actions"] = [
        {
            "name": "refund_invoice",
            "arguments": {"invoice_id": "inv-1"},
            "result": {"ok": False, "error_type": "Blocked"},
        }
    ]
    assert next_data.label_state(state, metadata()) == "refund"
    state["prior_actions"][0]["result"] = {
        "ok": True,
        "refund": {"invoice_id": "inv-1", "success": True},
    }
    assert next_data.label_state(state, metadata()) == "no_action"
    state["observation"] = None
    assert next_data.label_state(state, metadata()) == "defer"
    state["observation"] = {"ok": False, "error_type": "InvoiceNotFoundError"}
    assert next_data.label_state(state, metadata()) == "no_action"


def test_rollout_escalation_is_observable_and_duplicate_safe():
    state = observed("disputed")
    expected = metadata("escalate_dispute")
    assert next_data.label_state(state, expected) == "escalate"
    del state["observation"]["escalation_history"]
    assert next_data.label_state(state, expected) == "defer"
    state = observed("disputed", [{"id": "existing", "invoice_id": "inv-1"}])
    assert next_data.label_state(state, expected) == "no_action"
    assert next_data.label_state(observed("disputed"), metadata()) == "defer"


@pytest.mark.asyncio
async def test_freeze_then_collect_actual_dev_teacher_and_never_default_test(
    tmp_path, monkeypatch, authored_prior_root
):
    monkeypatch.setattr(next_data, "ROOT", authored_prior_root)
    monkeypatch.setattr(
        next_data, "__file__", str(authored_prior_root / "experiments/decision_model/next_data.py")
    )
    corpus = tmp_path / "corpus"
    manifest = await next_data.freeze(corpus)
    assert manifest["historical_official_dev_test_used"] == 0
    assert manifest["historical_official_train_used"] == 151
    assert len(manifest["historical_exclusions"]) == 9
    test_before = (corpus / "test-manifest.json").read_bytes()
    for split in next_data.SPLITS:
        selection = [
            json.loads(line)
            for line in (corpus / f"{split}-selection.jsonl").read_text().splitlines()
        ]
        assert all(set(row) == {"task_id", "group_id", "split"} for row in selection)
    output = tmp_path / "data"
    result = await next_data.collect(corpus, output, ("dev",))
    assert result["verified_teacher_tasks"] == 12
    assert result["final_test_collected"] is False
    assert result["rows"] == 71
    assert result["synthetic_missing_and_completed_fraction"] <= 0.20
    assert not (output / "test.jsonl").exists()
    assert (corpus / "test-manifest.json").read_bytes() == test_before
    rows = [json.loads(line) for line in (output / "dataset.jsonl").read_text().splitlines()]
    for row in rows:
        assert row["state"] == data.serialize_state(row["input_state"])
        assert set(row["input_state"]) == {
            "request",
            "target_invoice",
            "observation",
            "prior_actions",
            "policy",
            "authorization",
        }
        from experiments.decision_model.authorization import SERIALIZATION_VERSION

        assert row["serialization_version"] == SERIALIZATION_VERSION
        assert row["options"] == data.OPTIONS
        assert row["instructions"] == data.QUESTION
        assert data.ACTIONS[row["label"]] == row["label_action"]
        assert "expected_operations" not in row["state"]
        assert "defer_targets" not in row["state"]
    demonstrations = [
        json.loads(line) for line in (output / "demonstrations.jsonl").read_text().splitlines()
    ]
    assert all(
        d["complete"] and d["reward"] == 1 and d["verifier_components"]["reward"] == 1
        for d in demonstrations
    )
    assert all(d["tool_observations"] for d in demonstrations)
    with pytest.raises(FileExistsError):
        await next_data.freeze(corpus)
    with pytest.raises(FileExistsError):
        await next_data.collect(corpus, output, ("dev",))
