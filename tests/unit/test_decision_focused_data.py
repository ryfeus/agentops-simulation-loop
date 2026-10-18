"""Focused continuation examples stay observation-only and split by authored root."""

import itertools
import json
from collections import Counter

import pytest

from experiments.decision_model import data, focused_data, next_data


def test_focused_split_design_has_no_historical_or_test_examples():
    planned = focused_data.plans()
    focused_data.assert_disjoint(planned)
    assert Counter(p["split"] for p, _, _ in planned) == {"train": 144, "dev": 8, "calibration": 8}
    assert Counter(p["split"] for p in focused_data.PROTOTYPES) == {
        "train": 12,
        "dev": 4,
        "calibration": 4,
    }
    assert all(task.id.startswith("next-focused-") for _, task, _ in planned)
    assert all(
        not task.initial_state.refunds
        or all(
            i.status == "refunded"
            for i in task.initial_state.invoices
            if i.id in {r.invoice_id for r in task.initial_state.refunds}
        )
        for _, task, _ in planned
    )


def test_all_option_permutations_keep_semantic_pointer_label():
    for action in data.ACTIONS:
        original = {
            "options": data.OPTIONS,
            "label": data.ACTIONS.index(action),
            "label_action": action,
        }
        for order in itertools.permutations(range(4)):
            row = focused_data.permute_options(original, order)
            assert row["options"][row["label"]][0] == action
        assert original["options"] == data.OPTIONS
    with pytest.raises(ValueError, match="permutation"):
        focused_data.permute_options(original, (0, 0, 1, 2))


@pytest.mark.asyncio
async def test_actual_teacher_labels_ignore_parser_authorization_and_have_no_gold_in_input(
    tmp_path,
):
    from agentops_demo.taskify.harbor_renderer import render_harbor_task
    from rl.phase8b.execution_task import derive_execution_task

    prototype = next(p for p in focused_data.PROTOTYPES if "reviewed-dispute" in p["id"])
    task, defers = focused_data.scenario(prototype, 0)
    await render_harbor_task(task, tmp_path / "canonical")
    execution = derive_execution_task(tmp_path / "canonical", tmp_path / "task")
    metadata = next_data.metadata_for(task, "train", prototype["id"], defers, "new-authored")
    metadata["task_path"] = (
        str(execution.task_path) if hasattr(execution, "task_path") else str(tmp_path / "task")
    )
    temporary = tmp_path / "teacher"
    temporary.mkdir()
    rows, extras, demo = await next_data.teacher_rows(metadata, temporary)
    assert demo["verifier_components"]["reward"] == 1
    assert Counter(row["label_action"] for row in rows) == {
        "refund": 1,
        "escalate": 1,
        "no_action": 1,
    }
    for row in rows + extras:
        assert row["state"] == data.serialize_state(row["input_state"])
        assert "expected_operations" not in row["state"]
        assert "defer_targets" not in row["state"]
        row["input_state"]["authorization"] = {"status": "denied", "allowed_actions": []}
        assert next_data.label_state(row["input_state"], metadata) == row["label_action"]
    assert all(
        row["label_action"] == "defer" for row in extras if row["snapshot"] == "diagnostic_missing"
    )


@pytest.mark.asyncio
async def test_collection_rejects_test_split_before_creating_output(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cohort-manifest.json").write_text(json.dumps({"tasks": {"bad": {"split": "test"}}}))
    with pytest.raises(ValueError, match="Test/unknown"):
        await focused_data.collect(corpus, tmp_path / "data")
    assert not (tmp_path / "data").exists()
