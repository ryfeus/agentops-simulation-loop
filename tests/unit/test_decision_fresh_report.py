"""Integrity and paired comparison tests using synthetic outcomes only."""

from __future__ import annotations

import json

import pytest

from experiments.decision_model.fresh_report import ARMS, analyze, markdown
from experiments.decision_model.next_evaluate import digest, task_outcomes


@pytest.fixture
def run(tmp_path):
    tasks = {
        identifier: {
            "task_id": identifier,
            "split": "test",
            "composition_root": "synthetic-family",
            "expected_operations": {"invoice": "escalate_dispute"},
        }
        for identifier in ("fixture-a", "fixture-b")
    }
    cohort = tmp_path / "cohort-manifest.json"
    cohort.write_text(json.dumps({"tasks": tasks}))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "error": None,
                "task_ids": list(tasks),
                "repeats": 2,
                "planned_trials": 16,
                "cohort_previously_exposed": False,
                "cohort_manifest_sha256": digest(cohort),
            }
        )
    )
    rows = []
    for arm in ARMS:
        for identifier, task in tasks.items():
            for attempt in range(2):
                success = not (
                    (arm == "released_old" and identifier == "fixture-a" and attempt == 1)
                    or (arm == "llm_new" and identifier == "fixture-b" and attempt == 0)
                )
                row = {
                    "task_id": identifier,
                    "arm": arm,
                    "attempt": attempt,
                    "policy_completed": success,
                    "verifier": {"rewards": {"reward": int(success)}},
                    "tool_observations": [
                        {
                            "name": "escalate_dispute",
                            "arguments": {"invoice_id": "invoice"},
                            "result": {"ok": True},
                        }
                    ]
                    if success
                    else [],
                    "llm_calls": 1,
                    "llm_input_tokens": 20,
                    "llm_output_tokens": 10,
                    "end_to_end_policy_ms": 5,
                    "total_wall_ms": 10,
                }
                row.update(task_outcomes(row, task))
                rows.append(row)
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path


def test_report_pairs_runtime_within_policy_and_keeps_family_denominators(run):
    report = analyze(run)
    assert report["unique_tasks"] == 2
    assert report["composition_roots"] == 1
    assert report["total_trials"] == 16
    assert report["new_vs_old"]["released"]["improved_trial_pairs"] == 1
    assert report["new_vs_old"]["released"]["regressed_trial_pairs"] == 0
    assert report["new_vs_old"]["llm"]["regressed_trial_pairs"] == 1
    assert report["new_vs_old"]["llm"]["per_family_completion_delta"] == {"synthetic-family": -1}
    assert report["arms"]["released_old"]["tasks_completed_both_attempts"] == 1
    assert report["arms"]["released_new"]["tasks_completed_both_attempts"] == 2
    text = markdown(report)
    assert "2 held-out tasks across 1 composition roots" in text
    assert "does not demonstrate a learned improvement" in text
    assert "Released + repaired fallback" in text
    assert "selected" not in text.lower()


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate",
        "missing",
        "outcome",
        "missing_details",
        "incomplete",
        "error",
        "exposed",
        "freshness_unspecified",
        "planned_count",
        "repeats",
        "unknown_arm",
        "changed_cohort",
        "incomplete_success",
        "indeterminate_success",
    ],
)
def test_report_rejects_incomplete_unpaired_or_misrepresented_evidence(run, corruption):
    path = run / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "missing":
        rows.pop()
    elif corruption == "outcome":
        rows[0]["successful_unwanted_writes"] = 1
    elif corruption == "missing_details":
        rows[1]["missing_required_operation_details"] = []
    elif corruption == "incomplete":
        manifest["complete"] = False
    elif corruption == "error":
        manifest["error"] = "bounded run timed out"
    elif corruption == "exposed":
        manifest["cohort_previously_exposed"] = True
    elif corruption == "freshness_unspecified":
        del manifest["cohort_previously_exposed"]
    elif corruption == "planned_count":
        manifest["planned_trials"] = 15
    elif corruption == "repeats":
        manifest["repeats"] = 1
    elif corruption == "unknown_arm":
        rows[0]["arm"] = "tuned"
    elif corruption == "changed_cohort":
        cohort = run / "cohort-manifest.json"
        cohort.write_text(cohort.read_text() + "\n")
    elif corruption == "incomplete_success":
        rows[0]["tool_evidence_complete"] = False
    else:
        rows[0]["indeterminate_tool_calls"] = [{"name": "escalate_dispute"}]
    manifest_path.write_text(json.dumps(manifest))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        analyze(run)


def test_incomplete_failed_trial_is_retained_in_metrics(run):
    path = run / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["tool_evidence_complete"] = False
    rows[1]["indeterminate_tool_calls"] = [{"name": "escalate_dispute"}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = analyze(run)
    assert report["arms"]["released_old"]["incomplete_tool_evidence_tasks"] == 1
    assert report["arms"]["released_old"]["indeterminate_write_call_count"] == 1


def test_manifest_arm_contracts_are_retained_for_audit(run):
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["arm_contracts"] = {
        arm: {"runtime": "old" if arm.endswith("old") else "new"} for arm in ARMS
    }
    path.write_text(json.dumps(manifest))
    assert analyze(run)["execution"]["arm_contracts"] == manifest["arm_contracts"]
