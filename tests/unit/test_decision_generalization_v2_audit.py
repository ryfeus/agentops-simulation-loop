"""Audit integrity tests use synthetic retained files, never policies or held-out cases."""

import importlib.util
import json
from pathlib import Path

import pytest

from experiments.decision_model.next_evaluate import digest, task_outcomes


@pytest.fixture
def audit_module():
    path = Path("reports/decision-model-generalization-v2/audit.py")
    spec = importlib.util.spec_from_file_location("v2_audit_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_named_reference_documents_are_excluded_from_artifact_comparison(audit_module):
    prior = {
        "artifact_sha256": {"weights": "w", "calibration": "c", "old-reference": "r"},
        "baseline_reference_freeze_path": "old-reference",
    }
    current = {
        "artifact_sha256": {
            "weights": "w",
            "calibration": "c",
            "prior-reference": "p",
            "runtime-lock": "l",
        },
        "baseline_reference_freeze_path": "prior-reference",
        "runtime_lock_path": "runtime-lock",
    }
    assert audit_module.model_and_execution_artifacts(
        prior
    ) == audit_module.model_and_execution_artifacts(current)
    current["artifact_sha256"]["calibration"] = "changed"
    assert audit_module.model_and_execution_artifacts(
        prior
    ) != audit_module.model_and_execution_artifacts(current)
    current["artifact_sha256"]["calibration"] = "c"
    current["artifact_sha256"]["unrelated-extra"] = "e"
    assert audit_module.model_and_execution_artifacts(
        prior
    ) != audit_module.model_and_execution_artifacts(current)


@pytest.fixture
def regression_fixture(tmp_path, audit_module, monkeypatch):
    paid = tmp_path / "regression-paid"
    paid.mkdir()
    tasks = {
        f"offline-{family}-{variant}": {
            "task_id": f"offline-{family}-{variant}",
            "split": "test",
            "composition_root": f"offline-family-{family}",
            "expected_operations": {"fixture-invoice": "refund_invoice"},
        }
        for family in range(6)
        for variant in range(2)
    }
    audit_module.write(
        paid / "cohort-manifest.json", {"cohort_id": "billing-generalization-v1", "tasks": tasks}
    )
    audit_module.write(paid / "freeze.json", {"synthetic": True})
    freeze_checks = []
    monkeypatch.setattr(
        audit_module,
        "check_final_freeze",
        lambda freeze, cohort: freeze_checks.append((freeze, cohort)),
    )
    rows, order = [], []
    for task_index, (task_id, meta) in enumerate(tasks.items()):
        arms = ("released", "llm") if task_index % 2 == 0 else ("llm", "released")
        for arm in arms:
            row = {
                "task_id": task_id,
                "arm": arm,
                "attempt": 0,
                "policy_completed": True,
                "verifier": {"rewards": {"reward": 1}, "diagnostics": {}},
                "tool_observations": [
                    {
                        "name": "refund_invoice",
                        "arguments": {"invoice_id": "fixture-invoice"},
                        "result": {
                            "ok": True,
                            "refund": {"success": True, "invoice_id": "fixture-invoice"},
                        },
                    }
                ],
                "tool_evidence_complete": True,
                "indeterminate_tool_calls": [],
                "invoice_scope_blocked_attempts": 0,
                "cleanup": "deleted-owned-environment",
                "tool_call_count": 2,
                "llm_calls": 1,
                "llm_input_tokens": 20,
                "llm_output_tokens": 10,
                "end_to_end_policy_ms": 5,
                "total_wall_ms": 10,
            }
            row.update(task_outcomes(row, meta))
            position = len(rows)
            trial = paid / "trials" / f"{position:03d}-{arm}-{task_id}"
            native = trial / task_id / "verifier"
            native.mkdir(parents=True)
            (trial / "results.jsonl").write_text(json.dumps(row) + "\n")
            audit_module.write(native / "reward.json", row["verifier"]["rewards"])
            audit_module.write(native / "diagnostics.json", {})
            rows.append({**row, "execution_position": position})
            order.append({"task_id": task_id, "arm": arm, "attempt": 0, "position": position})
    (paid / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    audit_module.write(
        paid / "manifest.json",
        {
            "complete": True,
            "error": None,
            "cohort_previously_exposed": True,
            "planned_trials": 24,
            "max_job_seconds": 900,
            "training_steps": 0,
            "calibration_fits": 0,
            "execution_order": order,
            "run_total_wall_ms": 250,
        },
    )
    preflight = tmp_path / "preflight.json"
    audit_module.write(
        preflight,
        {
            arm: {
                "tasks": 24,
                "completed_and_verified": 24,
                "required_operations": 70,
                "successful_required_operations": 70,
                "successful_unwanted_writes": 0,
                "execution_errors": 0,
            }
            for arm in ["rule", "forced_observed_fallback"]
        },
    )
    evidence = {
        str(path.resolve()): digest(path)
        for path in [preflight, paid / "manifest.json", paid / "results.jsonl"]
    }
    return tmp_path, evidence, freeze_checks


def test_synthetic_paid_regression_evidence_is_independently_audited(
    audit_module, regression_fixture
):
    root, evidence, freeze_checks = regression_fixture
    result = audit_module.audit_regression(root, evidence)
    assert result["failures"] == []
    assert result["native_evidence_files_verified"] == 72
    assert result["retained_final_trials"] == result["owned_environments_deleted"] == 24
    assert result["usage"] == {"llm_calls": 24, "llm_input_tokens": 480, "llm_output_tokens": 240}
    assert len(freeze_checks) == 1


@pytest.mark.parametrize(
    "corruption",
    ["native", "preflight", "missing_lock_binding", "budget", "missing_pair", "cleanup"],
)
def test_synthetic_regression_corruption_cannot_pass(audit_module, regression_fixture, corruption):
    root, evidence, _ = regression_fixture
    paid = root / "regression-paid"
    if corruption == "native":
        target = next((paid / "trials").glob("*/*/verifier/reward.json"))
        audit_module.write(target, {"reward": 0})
    elif corruption == "preflight":
        target = root / "preflight.json"
        contents = audit_module.read(target)
        contents["rule"]["successful_required_operations"] = 69
        audit_module.write(target, contents)
    elif corruption == "missing_lock_binding":
        evidence.pop(str((paid / "results.jsonl").resolve()))
    elif corruption == "budget":
        target = paid / "manifest.json"
        contents = audit_module.read(target)
        contents["max_job_seconds"] = 3600
        audit_module.write(target, contents)
    else:
        target = paid / "results.jsonl"
        rows = audit_module.records(target)
        if corruption == "missing_pair":
            rows.pop()
        else:
            rows[0]["cleanup"] = "failed: retained environment"
        target.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = audit_module.audit_regression(root, evidence)
    assert result["failures"]


def test_blocked_model_quality_failure_is_retained_without_failing_core_gate(
    audit_module, regression_fixture
):
    root, evidence, _ = regression_fixture
    paid = root / "regression-paid"
    path = paid / "results.jsonl"
    rows = audit_module.records(path)
    row = rows[0]
    row["policy_completed"] = False
    row["authorization_blocked_attempts"] = 1
    task = audit_module.read(paid / "cohort-manifest.json")["tasks"][row["task_id"]]
    row.update(task_outcomes(row, task))
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    trial = next((paid / "trials").glob("000-*/results.jsonl"))
    trial.write_text(json.dumps({k: v for k, v in row.items() if k != "execution_position"}) + "\n")
    evidence[str(path.resolve())] = digest(path)
    result = audit_module.audit_regression(root, evidence)
    assert result["failures"] == []
    assert result["core_gate_passed"] is True
    assert result["strict_quality_passed"] is False
    assert result["policy_failure_trials"] == 1
    assert len(result["strict_quality_failure_details"]) == 1


@pytest.mark.parametrize("kind", ["argument", "graph"])
def test_regression_argument_and_graph_errors_fail_core_gate(
    audit_module, regression_fixture, kind
):
    root, evidence, _ = regression_fixture
    paid = root / "regression-paid"
    path = paid / "results.jsonl"
    rows = audit_module.records(path)
    row = rows[0]
    if kind == "argument":
        row["tool_observations"].append(
            {
                "name": "get_invoice",
                "arguments": {},
                "result": {"ok": False, "error_type": "InvoiceToolArgumentsError"},
            }
        )
    else:
        row["decisions"] = [{"fallback_result": {"graph_tool_errors": ["wrong arguments"]}}]
    task = audit_module.read(paid / "cohort-manifest.json")["tasks"][row["task_id"]]
    row.update(task_outcomes(row, task))
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    trial = next((paid / "trials").glob("000-*/results.jsonl"))
    trial.write_text(json.dumps({k: v for k, v in row.items() if k != "execution_position"}) + "\n")
    evidence[str(path.resolve())] = digest(path)
    result = audit_module.audit_regression(root, evidence)
    assert result["core_gate_passed"] is False
    assert len(result["argument_violation_trials"]) == 1
    assert result["failures"]
