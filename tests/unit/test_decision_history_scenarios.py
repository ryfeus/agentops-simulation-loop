"""New challenge intent is frozen and native-verifier satisfiable, independent of policies."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from agentops_demo.billing.service import BillingService
from agentops_demo.billing.sqlite_repository import SQLiteBillingRepository, initialize_database
from experiments.decision_model.history_scenarios import (
    PROTOTYPES,
    ROOT,
    build,
    normalized_instruction,
    novelty_audit,
    scenarios,
    structural_signature,
)
from rl.phase12.demonstrations import teacher_calls


@pytest.fixture(scope="module")
def cohort():
    return scenarios()


def test_fixed_eight_compositions_and_presentation_variants(cohort):
    assert len(cohort) == 24
    groups = Counter(s.benchmark.generation.prototype_id for s in cohort.values())
    assert len(groups) == 8 and set(groups.values()) == {3}
    assert len({structural_signature(s) for s in cohort.values()}) == 8
    assert len({normalized_instruction(s.instruction) for s in cohort.values()}) == 8
    assert len({s.id for s in cohort.values()}) == 24
    assert [s.model_dump() for s in cohort.values()] == [
        s.model_dump() for s in scenarios().values()
    ]


def test_no_historical_structure_overlap(cohort):
    audit = novelty_audit(cohort)
    assert audit["historical_tasks_compared"] == 340
    assert audit["structural_collisions"] == audit["normalized_instruction_collisions"] == 0
    assert audit["unique_new_structures"] == 8


def test_history_is_operation_specific_and_status_drift_is_explicit(cohort):
    by_id = cohort["history-v2-paid-review-cross-operation-v00"]
    escalated = {r.invoice_id for r in by_id.initial_state.escalations}
    refunds = {i.invoice_id for i in by_id.expected_invariants if i.type == "must_refund"}
    assert refunds & escalated  # An old review must not suppress a new refund.
    drift = cohort["history-v2-refund-ledger-status-drift-control-v00"]
    refunded = {r.invoice_id for r in drift.initial_state.refunds}
    protected = {i.invoice_id for i in drift.expected_invariants if i.type == "unchanged"}
    assert refunded <= protected
    assert any(i.id in refunded and i.status == "paid" for i in drift.initial_state.invoices)
    assert sum(not s["primary_history_probe"] for s in PROTOTYPES) == 1


def test_missing_targets_remain_required_and_unwritable(cohort):
    task = cohort["history-v2-missing-target-independent-review-v00"]
    existing = {i.id for i in task.initial_state.invoices}
    missing = set(task.benchmark.trajectory.required_inspections) - existing
    assert len(missing) == 1
    assert not missing & set(task.benchmark.mutation_policy.allowed_targets)
    assert any(i.type == "must_escalate" for i in task.expected_invariants)


def test_seed_data_contains_no_postconditions(cohort):
    for task in cohort.values():
        refunded = {r.invoice_id for r in task.initial_state.refunds}
        escalated = {r.invoice_id for r in task.initial_state.escalations}
        expected_writes = set()
        for invariant in task.expected_invariants:
            if invariant.type == "must_refund":
                assert invariant.invoice_id not in refunded
                expected_writes.add(invariant.invoice_id)
            elif invariant.type == "must_escalate":
                assert invariant.invoice_id not in escalated
                expected_writes.add(invariant.invoice_id)
        assert expected_writes == set(task.benchmark.mutation_policy.allowed_targets)
        assert len(teacher_calls(task)) <= 16


def native_verifier():
    path = ROOT / "benchmarks/disputed-refund/tests/verify.py"
    spec = importlib.util.spec_from_file_location("history_challenge_native_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task_id", list(scenarios()))
@pytest.mark.asyncio
async def test_gold_native_tools_satisfy_unmodified_harbor_verifier(task_id, tmp_path, cohort):
    task = cohort[task_id]
    database = tmp_path / "billing.db"
    await initialize_database(database, task.initial_state)
    service = BillingService(SQLiteBillingRepository(database), policy_mode="enforced")
    calls = teacher_calls(task)
    for call in calls:
        result = await getattr(service, call["name"])(**call["arguments"])
        if call["name"] != "get_invoice":
            assert result.success is True
    spec = tmp_path / "scenario.json"
    spec.write_text(json.dumps(task.model_dump(mode="json")))
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(json.dumps({"tool_calls": calls}))
    verifier = native_verifier()
    verifier.CONTAINER_SCENARIO_PATH = spec
    verifier.AGENT_RESULT_PATH = trajectory
    rewards, diagnostics = verifier.evaluate_database(database)
    assert rewards["reward"] == 1, diagnostics
    # An extra write ATTEMPT to a protected invoice must fail trajectory, even
    # if no data change occurs. This validates the negative-intent boundary.
    protected = next(i.invoice_id for i in task.expected_invariants if i.type == "unchanged")
    bad = [
        *calls,
        {"name": "refund_invoice", "arguments": {"invoice_id": protected, "reason": "No"}},
    ]
    trajectory.write_text(json.dumps({"tool_calls": bad}))
    rewards, diagnostics = verifier.evaluate_database(database)
    assert rewards["trajectory"] == rewards["reward"] == 0


@pytest.mark.asyncio
async def test_rendered_cohort_frozen_with_native_verifier_and_selection_only_records(tmp_path):
    output = tmp_path / "cohort"
    manifest = await build(output)
    assert manifest["task_count"] == 24
    assert manifest["primary_history_probe_tasks"] == 21
    expected_verifier = (ROOT / "benchmarks/disputed-refund/tests/verify.py").read_bytes()
    for metadata in manifest["tasks"].values():
        assert (Path(metadata["task_path"]) / "tests/verify.py").read_bytes() == expected_verifier
        assert len(metadata["seed_sql_sha256"]) == 64
    selection = [json.loads(line) for line in (output / "selection.jsonl").read_text().splitlines()]
    assert len(selection) == 24
    assert all(set(row) == {"task_id", "group_id", "split"} for row in selection)
    with pytest.raises(FileExistsError, match="Frozen"):
        await build(output)
