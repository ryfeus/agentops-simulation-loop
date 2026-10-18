"""Prospective family holdout authoring never evaluates a policy or model."""

import json
from collections import Counter

import pytest

from agentops_demo.taskify.integrity import scenario_sha256
from agentops_demo.validation.scenario import load_scenario
from experiments.decision_model import fresh_holdout


def test_fresh_roots_disjoint_from_all_prior_splits_and_stable_across_variants():
    current = fresh_holdout.scenarios()
    audit = fresh_holdout.novelty_audit(current)
    assert len(current) == 12
    assert audit["unique_prior_tasks"] == 412
    assert audit["unique_prior_roots"] == 74
    assert audit["unique_new_structures"] == 6
    assert audit["root_collisions"] == audit["structural_collisions"] == 0
    assert audit["normalized_instruction_collisions"] == 0
    for prototype in fresh_holdout.PROTOTYPES:
        variants = [fresh_holdout.make_scenario(prototype, v) for v in range(2)]
        assert len({fresh_holdout.structural_signature(t) for t in variants}) == 1
        assert len({fresh_holdout.normalized_instruction(t.instruction) for t in variants}) == 1
        assert variants[0].benchmark.trajectory.required_inspections != (
            variants[1].benchmark.trajectory.required_inspections
        )
        assert all(
            len(t.benchmark.trajectory.required_inspections) + len(fresh_holdout.operations(t))
            <= 16
            for t in variants
        )
    with pytest.raises(ValueError, match="outside frozen design"):
        fresh_holdout.make_scenario(fresh_holdout.PROTOTYPES[0], 2)


def test_collision_audit_rejects_old_family_even_when_renamed():
    from experiments.decision_model.history_scenarios import scenarios

    original = next(iter(scenarios().values()))
    renamed = original.model_copy(update={"id": "fresh-renamed-old-case"})
    with pytest.raises(ValueError, match="overlap"):
        fresh_holdout.novelty_audit({renamed.id: renamed})


@pytest.mark.asyncio
async def test_freeze_renderer_and_native_expectations_without_policy_or_teacher(
    tmp_path, monkeypatch
):
    from experiments.decision_model import authorization, next_data

    def forbidden(*args, **kwargs):
        raise AssertionError("Prospective holdout must never invoke policy or teacher")

    monkeypatch.setattr(authorization, "scoped_rule_action", forbidden)
    monkeypatch.setattr(authorization, "make_state", forbidden)
    monkeypatch.setattr(next_data, "teacher_rows", forbidden)
    output = tmp_path / "holdout"
    manifest = await fresh_holdout.freeze(output)
    assert manifest["cohort_previously_exposed"] is False
    assert manifest["pre_freeze_policy_rollouts"] == manifest["pre_freeze_model_inferences"] == 0
    assert manifest["split_task_counts"] == {"test": 12}
    assert manifest["split_root_counts"] == {"test": 6}
    assert manifest["expected_operation_counts"] == {"escalate_dispute": 26, "refund_invoice": 12}
    expected_counts = Counter()
    for metadata in manifest["tasks"].values():
        scenario = load_scenario(fresh_holdout.Path(metadata["task_path"]) / "scenario.yaml")
        assert scenario_sha256(scenario) == metadata["scenario_sha256"]
        assert metadata["split"] == "test"
        assert metadata["expected_operations"] == fresh_holdout.operations(scenario)
        expected_counts.update(metadata["expected_operations"].values())
        assert not set(metadata["defer_targets"]) & set(metadata["expected_operations"])
    assert dict(expected_counts) == manifest["expected_operation_counts"]
    fixtures = json.loads((output / "fixture-validation.json").read_text())
    assert len(fixtures) == 12
    assert all(r["expected_endpoint_accepted"] and r["missing_writes_rejected"] for r in fixtures)
    assert all(r["fixture_only"] and not r["agent_or_policy_executed"] for r in fixtures)
    assert max(r["fixture_calls"] for r in fixtures) == 10
    selection = [
        json.loads(line) for line in (output / "test-selection.jsonl").read_text().splitlines()
    ]
    assert all(set(row) == {"task_id", "group_id", "split"} for row in selection)
    with pytest.raises(FileExistsError):
        await fresh_holdout.freeze(output)
