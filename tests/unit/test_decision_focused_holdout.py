"""Focused holdout fixtures never query authorization, policy, teacher, or model."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentops_demo.taskify.integrity import harbor_task_sha256, scenario_sha256
from agentops_demo.validation.scenario import load_scenario
from experiments.decision_model import focused_holdout as cohort


@pytest.fixture(autouse=True)
def use_authored_prior_corpora(monkeypatch, authored_prior_root):
    monkeypatch.setattr(cohort, "ROOT", authored_prior_root)
    monkeypatch.setattr(
        cohort,
        "__file__",
        str(authored_prior_root / "experiments/decision_model" / Path(cohort.__file__).name),
    )


def test_prior_families_and_behavioral_structures_are_disjoint():
    current = cohort.scenarios()
    audit = cohort.novelty_audit(current)
    assert len(current) == 12
    assert audit["unique_prior_tasks"] == 632
    assert audit["unique_prior_roots"] == 124
    assert audit["source_task_counts"]["generalization-v2"] == 12
    assert audit["source_task_counts"]["generalization-v3"] == 12
    assert audit["source_task_counts"]["next-v1-all-splits"] == 223
    assert audit["source_task_counts"]["focused-train"] == 144
    assert audit["source_task_counts"]["focused-dev"] == 8
    assert audit["source_task_counts"]["focused-calibration"] == 8
    assert audit["root_collisions"] == audit["structural_collisions"] == 0
    assert audit["normalized_instruction_collisions"] == 0
    assert audit["unique_new_structures"] == 6
    for prototype in cohort.PROTOTYPES:
        variants = [cohort.make_scenario(prototype, v) for v in range(2)]
        assert len({cohort.structural_signature(t) for t in variants}) == 1
        assert len({cohort.normalized_instruction(t.instruction) for t in variants}) == 1
        assert all(
            len(t.benchmark.trajectory.required_inspections) + len(cohort.operations(t)) <= 11
            for t in variants
        )
        assert all(t.benchmark.generation.generator == cohort.COHORT_ID for t in variants)
        assert all(t.benchmark.generation.prototype_id == prototype["id"] for t in variants)
    with pytest.raises(ValueError, match="outside frozen design"):
        cohort.make_scenario(cohort.PROTOTYPES[0], 2)


def test_latest_prior_family_cannot_become_new_by_renaming():
    from experiments.decision_model.generalization_v4_holdout import scenarios

    old = next(iter(scenarios().values()))
    payload = old.model_dump(mode="json")
    payload["id"] = "generalization-v4-renamed-old"
    payload["instruction"] += " Please provide a concise final summary."
    payload["benchmark"]["generation"].update(
        family_id="generalization-v4-renamed-family",
        prototype_id="generalization-v4-renamed-prototype",
        generator=cohort.COHORT_ID,
    )
    renamed = cohort.Scenario.model_validate(payload)
    assert cohort.structural_signature(renamed) == cohort.structural_signature(old)
    assert cohort.normalized_instruction(renamed.instruction) != cohort.normalized_instruction(
        old.instruction
    )
    with pytest.raises(ValueError, match="overlap"):
        cohort.novelty_audit({renamed.id: renamed})


def test_fixture_freeze_blocks_runtime_and_parser_imports(tmp_path, authored_prior_root):
    output = tmp_path / "prospective-corpus"
    # A clean subprocess catches accidental transitive imports as well as direct
    # imports. The historical next_data generator would trip this sentinel.
    script = """
import asyncio
import importlib.abc
import sys
from pathlib import Path
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = {
            "experiments.decision_model.authorization",
            "experiments.decision_model.harbor",
            "experiments.decision_model.next_data",
            "experiments.decision_model.focused_data",
            "experiments.decision_model.data",
            "experiments.decision_model.model",
            "experiments.decision_model.runtime",
            "experiments.decision_model.teacher",
        }
        if fullname in forbidden:
            raise AssertionError("Forbidden policy/service import: " + fullname)
sys.meta_path.insert(0, RejectRuntime())
from experiments.decision_model import focused_holdout as cohort
from agentops_demo.billing import service
def forbidden_service(*args, **kwargs):
    raise AssertionError("BillingService execution forbidden during fixture validation")
service.BillingService = forbidden_service
cohort.ROOT = Path(sys.argv[2])
cohort.__file__ = str(cohort.ROOT / "experiments/decision_model" / Path(cohort.__file__).name)
output = Path(sys.argv[1])
asyncio.run(cohort.freeze(output))
frozen = (output / "cohort-manifest.json").read_bytes()
try:
    asyncio.run(cohort.freeze(output))
except FileExistsError:
    pass
else:
    raise AssertionError("Existing corpus must be immutable")
assert (output / "cohort-manifest.json").read_bytes() == frozen
"""
    subprocess.run(
        [sys.executable, "-B", "-c", script, str(output), str(authored_prior_root)], check=True
    )
    manifest = json.loads((output / "cohort-manifest.json").read_text())
    assert manifest["cohort_id"] == "billing-focused-test-v1"
    assert manifest["cohort_previously_exposed"] is False
    assert manifest["pre_freeze_policy_rollouts"] == 0
    assert manifest["pre_freeze_authorization_parser_checks"] == 0
    assert manifest["pre_freeze_model_inferences"] == 0
    assert manifest["pre_freeze_teacher_collections"] == 0
    assert manifest["split_task_counts"] == {"test": 12}
    assert manifest["split_root_counts"] == {"test": 6}
    assert manifest["expected_operation_counts"] == {"refund_invoice": 12, "escalate_dispute": 16}
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
    assert {r["fixture_calls"] for r in fixtures} == {8, 9}
    selection = [
        json.loads(row) for row in (output / "test-selection.jsonl").read_text().splitlines()
    ]
    assert all(set(row) == {"task_id", "group_id", "split"} for row in selection)
    assert len(selection) == 12
    assert not (output / "demonstrations.jsonl").exists()
    assert not (output / "dataset.jsonl").exists()
