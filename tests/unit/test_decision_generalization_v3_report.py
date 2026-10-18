"""Synthetic retained-evidence checks; no policy or held-out inputs are evaluated."""

import importlib.util
import json
from pathlib import Path

import pytest

from experiments.decision_model.next_evaluate import digest, summarize, task_outcomes


def load(name):
    spec = importlib.util.spec_from_file_location(
        "v3_test_" + name, Path("reports/decision-model-generalization-v3") / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    audit = load("audit")
    monkeypatch.setattr(audit, "validate_task", lambda task: None)
    tasks = {
        f"synthetic-{i}": {
            "task_id": f"synthetic-{i}",
            "split": "test",
            "composition_root": "synthetic-family",
            "expected_operations": {"invoice-a": "refund_invoice"},
        }
        for i in range(2)
    }
    cohort = {"tasks": tasks, "cohort_previously_exposed": True}
    audit.write(tmp_path / "cohort-manifest.json", cohort)
    (tmp_path / "source-original.py").write_text("original source snapshot\n")
    artifact = tmp_path / "weights"
    artifact.write_text("fixed weights\n")
    freeze = {
        "task_ids": list(tasks),
        "planned_trials": 8,
        "repeats": 2,
        "training_steps": 0,
        "calibration_fits": 0,
        "cohort_previously_exposed": True,
        "cohort_manifest_sha256": digest(tmp_path / "cohort-manifest.json"),
        "max_job_seconds": 3600,
        "source_sha256": {"/not-current/original.py": digest(tmp_path / "source-original.py")},
        "artifact_sha256": {str(artifact): digest(artifact)},
        "arm_contracts": {"released": {"threshold": 0.7}, "llm": {"threshold": 0.0}},
    }
    audit.write(tmp_path / "freeze.json", freeze)
    order, rows = [], []
    for position, (task, arm, attempt) in enumerate(audit.planned_order(list(tasks))):
        row = {
            "task_id": task,
            "arm": arm,
            "attempt": attempt,
            "policy_completed": True,
            "invoice_local_fallback": True,
            "tool_evidence_complete": True,
            "tool_call_count": 1,
            "llm_calls": 1,
            "llm_input_tokens": 20,
            "llm_output_tokens": 10,
            "end_to_end_policy_ms": 5,
            "total_wall_ms": 10,
            "cleanup": "deleted-owned-environment",
            "verifier": {"rewards": {"reward": 1}, "diagnostics": {}},
            "tool_observations": [
                {
                    "name": "refund_invoice",
                    "arguments": {"invoice_id": "invoice-a"},
                    "result": {"ok": True},
                }
            ],
        }
        row.update(task_outcomes(row, tasks[task]))
        trial = tmp_path / "trials" / f"{position:03d}-{arm}-{attempt}-{task}"
        native = trial / task / "verifier"
        native.mkdir(parents=True)
        (trial / "results.jsonl").write_text(json.dumps(row) + "\n")
        audit.write(native / "reward.json", row["verifier"]["rewards"])
        audit.write(native / "diagnostics.json", {})
        rows.append({**row, "execution_position": position})
        order.append({"task_id": task, "arm": arm, "attempt": attempt, "position": position})
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {
        **freeze,
        "complete": True,
        "error": None,
        "device": "mps",
        "max_tool_calls": 16,
        "max_llm_calls": 16,
        "fallback_timeout": 120,
        "task_timeout": 180,
        "execution_order": order,
        "freeze_sha256": digest(tmp_path / "freeze.json"),
        "arms": {a: summarize([r for r in rows if r["arm"] == a]) for a in audit.ARMS},
    }
    audit.write(tmp_path / "manifest.json", manifest)
    return audit, tmp_path, rows, tasks


def test_exact_interleaved_retained_evidence_passes(fixture):
    audit, root, _, _ = fixture
    result = audit.audit_run(root, expected_tasks=2, exposed=True)
    assert result["failures"] == []
    assert result["strict_gate_passed"] is True
    assert result["native_evidence_files_verified"] == 24


@pytest.mark.parametrize(
    "damage",
    [
        "native",
        "source",
        "model",
        "order",
        "summary",
        "missing",
        "cleanup",
        "tool_count",
        "scope",
        "freeze",
    ],
)
def test_corrupt_evidence_never_passes(fixture, damage):
    audit, root, rows, _ = fixture
    if damage == "native":
        audit.write(next((root / "trials").glob("*/*/verifier/reward.json")), {"reward": 0})
    elif damage in {"source", "model"}:
        (root / ("weights" if damage == "model" else "source-original.py")).write_text("changed")
    elif damage in {"order", "summary", "freeze"}:
        m = audit.read(root / "manifest.json")
        if damage == "order":
            m["execution_order"] = list(reversed(m["execution_order"]))
        elif damage == "freeze":
            m["freeze_sha256"] = "wrong"
        else:
            m["arms"]["released"]["llm_calls"] = 999
        audit.write(root / "manifest.json", m)
    else:
        if damage == "missing":
            rows.pop()
        elif damage == "cleanup":
            rows[0]["cleanup"] = "deleted-unowned"
        elif damage == "tool_count":
            rows[0]["tool_call_count"] = 0
        else:
            rows[0]["invoice_scope_blocked_attempts"] = 1
        (root / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert audit.audit_run(root, expected_tasks=2, exposed=True)["passed"] is False


def test_policy_failure_blocks_fresh_gate_but_is_not_integrity_failure(fixture):
    audit, root, rows, tasks = fixture
    row = rows[0]
    row["policy_completed"] = False
    row.update(task_outcomes(row, tasks[row["task_id"]]))
    (root / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    trial = next((root / "trials").glob("000-*/results.jsonl"))
    trial.write_text(json.dumps({k: v for k, v in row.items() if k != "execution_position"}) + "\n")
    manifest = audit.read(root / "manifest.json")
    manifest["arms"] = {a: summarize([r for r in rows if r["arm"] == a]) for a in audit.ARMS}
    audit.write(root / "manifest.json", manifest)
    result = audit.audit_run(root, expected_tasks=2, exposed=True)
    assert result["passed"] is True
    assert result["strict_gate_passed"] is False
    assert len(result["strict_failure_trials"]) == 1


def test_guard_rejection_is_not_automatically_a_model_error():
    module = load("report")
    decision = {
        "target_invoice": "invoice-a",
        "proposal": {"action": "refund"},
        "state": {"authorization": {"allowed_actions": []}},
        "fallback_reason": "authorization_blocked",
        "prohibited_proposal": True,
        "invoice_resolution": {"status": "resolved"},
    }
    row = {
        "task_id": "synthetic",
        "arm": "released",
        "attempt": 0,
        "completed_and_verified": True,
        "decisions": [decision],
    }
    result = module.diagnose(
        [row], {"synthetic": {"expected_operations": {"invoice-a": "refund_invoice"}}}
    )
    assert result["event_category_counts"] == {"guard_rejected_proposal_routed_to_fallback": 1}
    assert result["guard_rejected_recovered_decisions"] == 1
    assert result["proposal_and_recovery_events"][0]["raw_decision"] == decision


def test_observed_duplicate_proposal_is_retained_even_when_recovered():
    module = load("report")
    decision = {
        "target_invoice": "invoice-a",
        "proposal": {"action": "escalate"},
        "state": {
            "observation": {
                "invoice": {"id": "invoice-a", "status": "paid"},
                "escalation_history": {
                    "invoice_id": "invoice-a",
                    "complete": True,
                    "records": [{"id": "review", "invoice_id": "invoice-a"}],
                },
            }
        },
        "invoice_resolution": {"status": "resolved"},
        "fallback_reason": "authorization_blocked",
    }
    row = {
        "task_id": "synthetic",
        "arm": "released",
        "attempt": 0,
        "completed_and_verified": True,
        "decisions": [decision],
    }
    result = module.diagnose([row], {"synthetic": {"expected_operations": {}}})
    assert (
        result["event_category_counts"][
            "model_duplicate_review_proposal_with_complete_observed_history"
        ]
        == 1
    )
    assert result["residual_trials"] == []
    decision["state"]["observation"]["escalation_history"]["complete"] = False
    result = module.diagnose([row], {"synthetic": {"expected_operations": {}}})
    assert (
        "model_duplicate_review_proposal_with_complete_observed_history"
        not in result["event_category_counts"]
    )


def test_report_pairs_and_families_recompute(fixture):
    _, root, _, _ = fixture
    result = load("report").analyze(root)
    assert result["paired_comparison"] == {"released_wins": 0, "released_losses": 0, "ties": 4}
    assert result["arms"]["released"]["successful_required_operations"] == 4
    assert result["distinct_families"] == 1
    assert result["resource_accounting"]["usage"]["llm_calls"] == {
        "known_total": 8,
        "missing_trials": 0,
    }


def test_historical_comparison_uses_only_matched_attempts(fixture, tmp_path):
    _, root, rows, tasks = fixture
    old = tmp_path / "historical"
    old.mkdir()
    rows = [{**r, "completed_and_verified": False} for r in rows if r["attempt"] == 0]
    (old / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (old / "cohort-manifest.json").write_text(json.dumps({"tasks": tasks}))
    result = load("report").historical_comparison(root, old)
    assert result["matched_trials"] == 4
    assert result["arms"]["released"]["wins"] == 2
    assert "provider variation" in result["caveat"]


def test_historical_audit_uses_retained_sources_and_native_evidence(fixture):
    audit, root, _, _ = fixture
    result = audit.audit_historical(root)
    assert result["passed"] is True
    assert result["manifest_hash_binding_available"] is True
    path = next((root / "trials").glob("*/*/verifier/diagnostics.json"))
    audit.write(path, {"changed": True})
    assert audit.audit_historical(root)["passed"] is False


@pytest.mark.parametrize(
    "after_ids,passed",
    [(["old-a", "old-b"], True), (["old-a"], False), (["old-a", "old-b", "new"], False)],
)
def test_container_inventory_requires_preexisting_preservation(tmp_path, after_ids, passed):
    audit = load("audit")
    before, after = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    before.write_text("".join(json.dumps({"ID": i}) + "\n" for i in ["old-a", "old-b"]))
    after.write_text("".join(json.dumps({"ID": i}) + "\n" for i in after_ids))
    assert audit.audit_container_inventories(before, after)["passed"] is passed


def test_fresh_claim_without_provenance_fails(fixture):
    audit, root, _, _ = fixture
    result = audit.audit_run(root, expected_tasks=2, exposed=False)
    assert result["passed"] is False
    assert "missing prospective first-access marker" in result["failures"]
