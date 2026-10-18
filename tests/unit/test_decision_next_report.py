"""Offline final report integrity; synthetic traces only, no held-out execution."""

from __future__ import annotations

import json

import pytest

from experiments.decision_model.next_evaluate import digest, task_outcomes
from experiments.decision_model.next_report import analyze, markdown


@pytest.fixture
def run(tmp_path):
    task = {
        "task_id": "fixture",
        "split": "test",
        "composition_root": "synthetic",
        "expected_operations": {"invoice": "refund_invoice"},
    }
    cohort = tmp_path / "cohort-manifest.json"
    cohort.write_text(json.dumps({"tasks": {"fixture": task}}))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "error": None,
                "task_ids": ["fixture"],
                "repeats": 2,
                "cohort_manifest_sha256": digest(cohort),
            }
        )
    )
    rows = []
    for arm in ("rule", "released", "tuned", "llm"):
        for attempt in range(2):
            success = not (arm == "released" and attempt == 1)
            row = {
                "task_id": "fixture",
                "arm": arm,
                "attempt": attempt,
                "policy_completed": success,
                "verifier": {"rewards": {"reward": int(success)}},
                "tool_observations": [
                    {
                        "name": "refund_invoice",
                        "arguments": {"invoice_id": "invoice"},
                        "result": {"ok": True},
                    }
                ]
                if success
                else [],
                "llm_calls": 0,
                "llm_input_tokens": 0,
                "llm_output_tokens": 0,
                "end_to_end_policy_ms": 5,
                "total_wall_ms": 10,
            }
            row.update(task_outcomes(row, task))
            rows.append(row)
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path


def test_final_report_pairs_attempts_and_does_not_count_them_as_independent_tasks(run):
    report = analyze(run)
    assert report["unique_tasks"] == report["composition_roots"] == 1
    assert report["total_trials"] == 8
    assert report["arms"]["released"]["tasks_completed_any_attempt"] == 1
    assert report["arms"]["released"]["tasks_completed_both_attempts"] == 0
    assert report["arms"]["released"]["missing_required_operations"] == 1
    assert report["tuned_vs_released"]["improved_trial_pairs"] == 1
    assert "1 held-out tasks" in markdown(report)


def test_exposed_cohort_and_selected_baseline_are_reported_without_training_gain(run):
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(
        cohort_previously_exposed=True,
        selected_checkpoint_role="released_baseline",
        invoice_local_fallback=True,
        max_llm_calls=16,
        fallback_timeout=120,
    )
    path.write_text(json.dumps(manifest))
    report = analyze(run)
    text = markdown(report)
    assert "previously exposed cohort" in text
    assert "selected epoch-0 released baseline" in text
    assert "does not demonstrate a learned improvement" in text
    assert "1 held-out tasks" not in text
    assert report["execution"]["invoice_local_fallback"] is True
    assert report["execution"]["max_llm_calls"] == 16


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "outcome", "incomplete"])
def test_final_report_rejects_partial_duplicate_or_inconsistent_evidence(run, corruption):
    path = run / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "missing":
        rows.pop()
    elif corruption == "outcome":
        rows[0]["successful_unwanted_writes"] = 1
    else:
        manifest = run / "manifest.json"
        payload = json.loads(manifest.read_text())
        payload["complete"] = False
        manifest.write_text(json.dumps(payload))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        analyze(run)
