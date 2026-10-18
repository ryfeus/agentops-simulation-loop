"""New prospective compositions are audited without executing authorization or policy."""

import json
from pathlib import Path

import pytest

from agentops_demo.taskify.integrity import harbor_task_sha256, scenario_sha256
from agentops_demo.validation.scenario import load_scenario
from experiments.decision_model import generalization_holdout as cohort


def test_all_previous_families_disjoint_and_variants_are_one_structure_per_root():
    current = cohort.scenarios()
    audit = cohort.novelty_audit(current)
    assert len(current) == 12
    assert audit["unique_prior_tasks"] == 424
    assert audit["unique_prior_roots"] == 80
    assert audit["source_task_counts"]["fallback-fresh"] == 12
    assert audit["root_collisions"] == 0
    assert audit["structural_collisions"] == 0
    assert audit["normalized_instruction_collisions"] == 0
    assert audit["unique_new_structures"] == 6
    for prototype in cohort.PROTOTYPES:
        variants = [cohort.make_scenario(prototype, v) for v in range(2)]
        assert len({cohort.structural_signature(t) for t in variants}) == 1
        assert len({cohort.normalized_instruction(t.instruction) for t in variants}) == 1
        assert all(
            len(t.benchmark.trajectory.required_inspections) + len(cohort.operations(t)) <= 16
            for t in variants
        )
        assert all(t.benchmark.generation.generator == cohort.COHORT_ID for t in variants)
        assert all(t.benchmark.generation.prototype_id == prototype["id"] for t in variants)
    with pytest.raises(ValueError, match="outside frozen design"):
        cohort.make_scenario(cohort.PROTOTYPES[0], 2)


def test_prior_fallback_family_cannot_become_new_by_renaming():
    from experiments.decision_model.fresh_holdout import scenarios

    old = next(iter(scenarios().values()))
    renamed = old.model_copy(update={"id": "generalization-renamed-old"})
    with pytest.raises(ValueError, match="overlap"):
        cohort.novelty_audit({renamed.id: renamed})


@pytest.mark.asyncio
async def test_native_endpoint_validation_and_freeze_never_execute_a_policy(tmp_path, monkeypatch):
    from agentops_demo.billing import service
    from experiments.decision_model import authorization, next_data

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "No authorization, policy, teacher or billing-tool execution permitted"
        )

    for method in ["authorization", "authorize_action", "scoped_rule_action", "make_state"]:
        monkeypatch.setattr(authorization, method, forbidden)
    monkeypatch.setattr(next_data, "teacher_rows", forbidden)
    monkeypatch.setattr(service, "BillingService", forbidden)
    output = tmp_path / "prospective-corpus"
    manifest = await cohort.freeze(output)
    assert manifest["cohort_id"] == "billing-generalization-v1"
    assert manifest["cohort_previously_exposed"] is False
    assert manifest["pre_freeze_policy_rollouts"] == 0
    assert manifest["pre_freeze_authorization_parser_checks"] == 0
    assert manifest["pre_freeze_model_inferences"] == 0
    assert manifest["pre_freeze_teacher_collections"] == 0
    assert manifest["split_task_counts"] == {"test": 12}
    assert manifest["split_root_counts"] == {"test": 6}
    assert manifest["expected_operation_counts"] == {"refund_invoice": 16, "escalate_dispute": 16}
    for meta in manifest["tasks"].values():
        task_path = Path(meta["task_path"])
        task = load_scenario(task_path / "scenario.yaml")
        assert scenario_sha256(task) == meta["scenario_sha256"]
        assert harbor_task_sha256(task_path) == meta["execution_task_sha256"]
        assert meta["expected_operations"] == cohort.operations(task)
        assert meta["split"] == "test"
        assert not set(meta["defer_targets"]) & set(meta["expected_operations"])
        assert "expected_operations" not in (task_path / "instruction.md").read_text()
        assert meta["author_intent"]
    fixtures = json.loads((output / "fixture-validation.json").read_text())
    assert len(fixtures) == 12
    assert all(r["expected_endpoint_accepted"] and r["missing_writes_rejected"] for r in fixtures)
    assert all(r["fixture_only"] and not r["agent_or_policy_executed"] for r in fixtures)
    assert {r["fixture_calls"] for r in fixtures} == {8, 9, 10}
    selection = [
        json.loads(row) for row in (output / "test-selection.jsonl").read_text().splitlines()
    ]
    assert all(set(row) == {"task_id", "group_id", "split"} for row in selection)
    assert len(selection) == 12
    assert not (output / "demonstrations.jsonl").exists()
    assert not (output / "dataset.jsonl").exists()
    frozen = (output / "cohort-manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        await cohort.freeze(output)
    assert (output / "cohort-manifest.json").read_bytes() == frozen
