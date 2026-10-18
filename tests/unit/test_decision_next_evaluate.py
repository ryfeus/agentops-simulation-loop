"""Real SQLite/native verifier checks; no AWS, Docker, or held-out test outcomes."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pytest
from conftest import GOLDEN_SCENARIO

from agentops_demo.contracts.scenario import Scenario
from agentops_demo.taskify.harbor_renderer import render_harbor_task
from agentops_demo.taskify.integrity import harbor_task_sha256, scenario_sha256
from agentops_demo.validation.scenario import load_scenario
from experiments.decision_model.next_evaluate import (
    build_final_freeze,
    check_final_freeze,
    evaluate_candidate,
    evaluate_tasks,
    final_comparison,
    load_tasks,
    summarize,
)


@pytest.fixture
async def task(tmp_path):
    payload = load_scenario(GOLDEN_SCENARIO).model_dump(mode="json")
    payload["id"] = "offline-paid-refund"
    payload["instruction"] = "Refund invoice inv-123."
    payload["initial_state"]["invoices"][0]["status"] = "paid"
    payload["expected_invariants"] = [
        {"type": "must_refund", "invoice_id": "inv-123"},
        {"type": "no_escalation", "invoice_id": "inv-123"},
    ]
    payload["benchmark"] = {
        "category": "billing",
        "archetype": "offline-test",
        "difficulty": "easy",
        "mutation_policy": {"allowed_targets": ["inv-123"]},
        "trajectory": {
            "required_inspections": ["inv-123"],
            "require_inspection_before_mutation": True,
        },
    }
    scenario = Scenario.model_validate(payload)
    path = await render_harbor_task(scenario, tmp_path / "native-task")
    return {
        "task_id": scenario.id,
        "split": "dev",
        "group_id": "family-a",
        "task_path": str(path),
        "scenario_sha256": scenario_sha256(scenario),
        "execution_task_sha256": harbor_task_sha256(path),
        "expected_operations": {"inv-123": "refund_invoice"},
    }


async def test_actual_service_native_verifier_and_observation_only_predictor(task, tmp_path):
    states = []

    def predict(state):
        states.append(state)
        assert "expected_operations" not in state
        assert "initial_state" not in state
        assert state["observation"]["invoice"]["status"] == "paid"
        assert state["observation"]["escalation_history"]["complete"] is True
        return {"action": "refund", "confidence": 1.0}

    result = await evaluate_tasks(predict, [task], tmp_path / "run")
    assert len(states) == 1
    assert result["completed_and_verified"] == 1
    assert result["successful_required_operations"] == 1
    assert result["missing_required_operations"] == 0
    assert result["successful_unwanted_writes"] == 0
    assert result["llm_calls"] == 0
    row = json.loads(Path(result["records_path"]).read_text())
    assert row["verifier"]["rewards"]["must_refund"] == 1
    assert row["verifier"]["rewards"]["trajectory"] == 1
    assert row["decisions"][0]["state"] == states[0]


async def test_defer_without_fallback_is_unresolved_and_missing_required_work(task, tmp_path):
    result = await evaluate_tasks(
        lambda state: {"action": "defer", "confidence": 1}, [task], tmp_path / "defer"
    )
    assert result["completed_and_verified"] == 0
    assert result["missing_required_operations"] == 1
    assert result["fallback_unresolved"] == 1
    assert result["llm_calls"] == 0


async def test_guarded_bad_proposal_is_scored_before_enforcement(task, tmp_path):
    result = await evaluate_tasks(
        lambda state: {"action": "escalate", "confidence": 1}, [task], tmp_path / "wrong"
    )
    assert result["successful_unwanted_writes"] == 0
    assert result["raw_prohibited_proposals"] == 1
    assert result["accepted_prohibited_proposals"] == 1
    assert result["authorization_blocked_attempts"] == 1
    assert result["completed_and_verified"] == 0
    assert result["missing_required_operations"] == 1


async def test_threshold_filters_raw_risk_but_remains_unresolved(task, tmp_path):
    result = await evaluate_tasks(
        lambda state: {"action": "escalate", "confidence": 0.5},
        [task],
        tmp_path / "threshold",
        threshold=0.9,
    )
    assert result["raw_prohibited_proposals"] == 1
    assert result["accepted_prohibited_proposals"] == 0
    assert result["authorization_blocked_attempts"] == 0
    assert result["completed_and_verified"] == 0


async def test_task_content_drift_fails_before_prediction(task, tmp_path):
    Path(task["task_path"], "instruction.md").write_text("changed")
    with pytest.raises(ValueError, match="task contents changed"):
        await evaluate_tasks(
            lambda state: pytest.fail("must not predict"), [task], tmp_path / "bad"
        )
    assert not (tmp_path / "bad").exists()


async def test_test_access_refused_before_prediction(task, tmp_path):
    task["split"] = "test"
    with pytest.raises(ValueError, match="may not inspect test"):
        await evaluate_tasks(
            lambda state: pytest.fail("must not predict"), [task], tmp_path / "test"
        )
    with pytest.raises(ValueError, match="test remains reserved"):
        await evaluate_candidate(tmp_path / "missing.json", None, tmp_path / "test2", split="test")


def test_selection_prioritizes_outcomes_and_marks_execution_errors_ineligible():
    base = {
        "completed_and_verified": True,
        "verifier_pass": True,
        "successful_unwanted_writes": 0,
        "missing_required_operations": 0,
        "successful_required_operations": 1,
        "required_operations": 1,
        "decisions": [],
        "prohibited_proposals": 0,
    }
    good = summarize([base])
    omitted = summarize(
        [{**base, "completed_and_verified": False, "missing_required_operations": 1}]
    )
    harmful = summarize([{**base, "successful_unwanted_writes": 1}])
    assert good["selection_key"] > omitted["selection_key"] > harmful["selection_key"]
    assert not summarize([{**base, "error": "fixture"}])["selection_eligible"]
    for decision in [
        {"proposal": {"error": "bad tensor"}, "fallback_reason": "model_error"},
        {"proposal": {}, "fallback_reason": "timeout"},
    ]:
        result = summarize([{**base, "decisions": [decision]}])
        assert result["inference_errors"] == 1
        assert not result["selection_eligible"]
    assert summarize(
        [
            {
                **base,
                "decisions": [{"proposal": {"action": "defer"}, "fallback_reason": "defer"}],
                "fallback_error": "fallback_unconfigured",
            }
        ]
    )["selection_eligible"]


def test_family_balanced_selection_only_reads_identities(tmp_path):
    manifest = tmp_path / "manifest.json"
    tasks = {
        f"{family}-{index}": {
            "task_id": f"{family}-{index}",
            "split": "dev",
            "composition_root": family,
            "task_path": "unused",
        }
        for family in ["a", "b", "c", "d"]
        for index in range(3)
    }
    manifest.write_text(json.dumps({"tasks": tasks}))
    selected = load_tasks(manifest, "dev", 8)
    assert [row["task_id"] for row in selected] == [
        "a-0",
        "b-0",
        "c-0",
        "d-0",
        "a-1",
        "b-1",
        "c-1",
        "d-1",
    ]


@pytest.fixture
def final_args(task, tmp_path):
    task = {**task, "split": "test"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": {task["task_id"]: task}}))
    paths = {}
    for name in ("released", "tuned"):
        path = tmp_path / name
        path.mkdir()
        (path / "adapter.safetensors").write_bytes(b"offline checkpoint identity fixture")
        paths[name] = path
    for name in (
        "released_calibration",
        "tuned_calibration",
        "fallback_config",
        "selection_manifest",
    ):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({"threshold": 0.9, "selection_complete": True}))
        paths[name] = path
    wheel = tmp_path / "fake.whl"
    wheel.write_bytes(b"offline")
    freeze = build_final_freeze(manifest, wheel=wheel, **paths)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze))
    return argparse.Namespace(
        manifest=manifest,
        freeze=freeze_path,
        output=tmp_path / "final",
        tasks_root=None,
        wheel=wheel,
        device="cpu",
        repeats=2,
        limit=1,
        max_job_seconds=1800,
        **{key: value for key, value in paths.items() if key != "selection_manifest"},
    )


def test_selection_marker_and_artifact_drift_prevent_final_access(final_args):
    freeze = json.loads(final_args.freeze.read_text())
    check_final_freeze(freeze, final_args.manifest)
    Path(freeze["selection_manifest"]).write_text(json.dumps({"selection_complete": False}))
    with pytest.raises(ValueError, match="not marked complete"):
        check_final_freeze(freeze, final_args.manifest)
    Path(freeze["selection_manifest"]).write_text(
        json.dumps({"selection_complete": True, "threshold": 0.9})
    )
    with pytest.raises(ValueError, match="artifact changed"):
        check_final_freeze(freeze, final_args.manifest)


@pytest.mark.parametrize("invoice_local", [False, True])
async def test_final_interleaves_arms_reuses_models_and_preserves_budgets(
    final_args, monkeypatch, invoice_local
):
    from experiments.decision_model import harbor, model, next_evaluate

    loads = []
    episodes = []
    fallbacks = []
    final_args.invoice_local_fallback = invoice_local
    final_args.max_job_seconds = 3600 if invoice_local else 1800
    final_args.cohort_previously_exposed = invoice_local
    final_args.selected_checkpoint_role = "released_baseline" if invoice_local else "unspecified"
    freeze = json.loads(final_args.freeze.read_text())
    freeze.update(
        invoice_local_fallback=invoice_local,
        cohort_previously_exposed=invoice_local,
        selected_checkpoint_role=final_args.selected_checkpoint_role,
    )
    final_args.freeze.write_text(json.dumps(freeze))

    class FakeModel:
        def __init__(self, checkpoint, **kwargs):
            loads.append(str(checkpoint))

        def predict(self, row):
            assert "expected_operations" not in json.loads(row["state"])
            return {"action": "no_action", "confidence": 1}

    def fallback(config, *, invoice_local_fallback=False):
        assert invoice_local_fallback is invoice_local
        value = object()
        fallbacks.append(value)
        return value

    async def fake_evaluate(predict, tasks, output, **kwargs):
        assert kwargs["fallback"] is fallbacks[0]
        assert kwargs["backend"] == "harbor"
        assert kwargs["max_tool_calls"] == 16
        assert kwargs["allow_test"] is True
        assert kwargs["task_timeout"] <= 180
        assert kwargs["invoice_local_fallback"] is invoice_local
        episodes.append((kwargs["arm"], kwargs["attempt"]))
        output.mkdir(parents=True)
        row = {
            "task_id": tasks[0]["task_id"],
            "arm": kwargs["arm"],
            "attempt": kwargs["attempt"],
            "completed_and_verified": True,
            "verifier_pass": True,
            "successful_unwanted_writes": 0,
            "missing_required_operations": 0,
            "successful_required_operations": 1,
            "required_operations": 1,
            "llm_calls": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
        }
        path = output / "results.jsonl"
        path.write_text(json.dumps(row) + "\n")
        return {"records_path": str(path)}

    monkeypatch.setattr(model, "DecisionModel", FakeModel)
    monkeypatch.setattr(harbor, "llm_fallback", fallback)
    monkeypatch.setattr(next_evaluate, "evaluate_tasks", fake_evaluate)
    for name in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "BILLING_MODEL_TEMPERATURE",
        "BILLING_MODEL_MAX_TOKENS",
        "BILLING_MODEL_TOTAL_ATTEMPTS",
    ):
        monkeypatch.setenv(name, "offline")
    result = await final_comparison(final_args)
    assert result["complete"]
    assert result["invoice_local_fallback"] is invoice_local
    assert result["cohort_previously_exposed"] is invoice_local
    assert result["selected_checkpoint_role"] == final_args.selected_checkpoint_role
    assert len(loads) == 2
    assert len(fallbacks) == 1
    assert episodes == [
        ("rule", 0),
        ("released", 0),
        ("tuned", 0),
        ("llm", 0),
        ("released", 1),
        ("tuned", 1),
        ("llm", 1),
        ("rule", 1),
    ]
    assert all(arm["tasks"] == 2 for arm in result["arms"].values())


@pytest.mark.parametrize(
    "key,value",
    [
        ("invoice_local_fallback", True),
        ("cohort_previously_exposed", True),
        ("selected_checkpoint_role", "released_baseline"),
    ],
)
async def test_final_rejects_unfrozen_policy_or_reporting_options(final_args, key, value):
    setattr(final_args, key, value)
    with pytest.raises(ValueError, match="frozen evaluation option"):
        await final_comparison(final_args)
    assert not final_args.output.exists()


@pytest.mark.parametrize("budget", [59, 3601])
async def test_final_rejects_unbounded_job_before_execution(final_args, budget):
    final_args.max_job_seconds = budget
    with pytest.raises(ValueError, match="60-3600"):
        await final_comparison(final_args)
    assert not final_args.output.exists()


async def test_evaluator_forwards_local_mode_and_records_shared_budgets(
    task, tmp_path, monkeypatch
):
    from experiments.decision_model import harbor

    original = harbor.run_episode

    async def observed_run(**kwargs):
        assert kwargs["invoice_local_fallback"] is True
        assert kwargs["max_llm_calls"] == 16
        assert kwargs["fallback_timeout"] == 120
        return await original(**kwargs)

    monkeypatch.setattr(harbor, "run_episode", observed_run)
    result = await evaluate_tasks(None, [task], tmp_path / "local", invoice_local_fallback=True)
    assert result["completed_and_verified"] == 1
    manifest = json.loads((tmp_path / "local/manifest.json").read_text())
    assert manifest["invoice_local_fallback"] is True
    assert manifest["max_tool_calls"] == manifest["max_llm_calls"] == 16
    assert manifest["fallback_timeout"] == 120


async def test_outer_timeout_keeps_completed_write_usage_and_runs_native_verifier(
    task, tmp_path, monkeypatch
):
    from experiments.decision_model import harbor

    # The actual service permits this paid refund, but this evaluator-only task
    # contract expects no write. A lost snapshot would falsely count zero harm.
    task = {**task, "expected_operations": {}}

    async def interrupted_episode(*, invoke, evidence_sink, **kwargs):
        observation = await invoke("get_invoice", {"invoice_id": "inv-123"})
        refund = await invoke("refund_invoice", {"invoice_id": "inv-123", "reason": "test"})
        assert observation["ok"] and refund["ok"]
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            evidence_sink.update(
                policy_completed=False,
                tool_observations=[
                    {
                        "name": "refund_invoice",
                        "arguments": {"invoice_id": "inv-123"},
                        "result": refund,
                        "prohibited_attempt": False,
                    }
                ],
                llm_calls=2,
                llm_input_tokens=19,
                llm_output_tokens=7,
                decisions=[],
                episode_cancelled=True,
                tool_evidence_complete=True,
            )
            raise

    monkeypatch.setattr(harbor, "run_episode", interrupted_episode)
    metrics = await evaluate_tasks(
        None,
        [task],
        tmp_path / "timeout-evidence",
        task_timeout=0.05,
        invoice_local_fallback=True,
    )
    row = json.loads(Path(metrics["records_path"]).read_text())
    assert row["error"].startswith("TimeoutError")
    assert row["episode_cancelled"] and not row["policy_completed"]
    assert row["successful_unwanted_writes"] == metrics["successful_unwanted_writes"] == 1
    assert row["verifier"]["rewards"]["reward"] == 1.0
    assert row["cleanup"] == "local-evidence-retained"
    assert metrics["llm_calls"] == 2
    assert metrics["llm_input_tokens"] == 19 and metrics["llm_output_tokens"] == 7
    assert not metrics["selection_eligible"]


def test_inflight_unknown_effect_is_not_hidden_by_zero_confirmed_write_count():
    record = {
        "completed_and_verified": False,
        "verifier_pass": False,
        "successful_unwanted_writes": 0,
        "missing_required_operations": 1,
        "successful_required_operations": 0,
        "required_operations": 1,
        "tool_evidence_complete": False,
        "indeterminate_tool_calls": [
            {"name": "refund_invoice", "outcome": "unknown_after_cancellation"}
        ],
    }
    metrics = summarize([record])
    assert metrics["incomplete_tool_evidence_tasks"] == 1
    assert metrics["indeterminate_tool_call_count"] == 1
    assert metrics["indeterminate_write_call_count"] == 1
    assert metrics["successful_unwanted_writes"] == 0  # Confirmed only; effect is unknown.
    assert not metrics["selection_eligible"]


async def test_repeated_native_verification_preserves_frozen_task_hash(task, tmp_path, monkeypatch):
    import sys

    # Ensure the fix works without the operational PYTHONDONTWRITEBYTECODE workaround.
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    path = Path(task["task_path"])
    original = task["execution_task_sha256"]
    assert harbor_task_sha256(path) == original
    for attempt in range(2):
        result = await evaluate_tasks(
            lambda state: {"action": "refund", "confidence": 1.0},
            [task],
            tmp_path / f"repeated-verifier-{attempt}",
        )
        assert result["completed_and_verified"] == 1
        assert result["successful_required_operations"] == 1
        assert harbor_task_sha256(path) == original
        assert not list(path.rglob("*.pyc"))


async def test_retained_episode_runner_injection_preserves_native_evidence(task, tmp_path):
    from experiments.decision_model.harbor import run_episode

    calls = []

    async def retained(**kwargs):
        calls.append(kwargs)
        return await run_episode(**kwargs)

    result = await evaluate_tasks(
        lambda state: {"action": "refund", "confidence": 1},
        [task],
        tmp_path / "retained-runtime",
        episode_runner=retained,
        invoice_local_fallback=True,
    )
    assert len(calls) == 1
    assert calls[0]["invoice_local_fallback"] is True
    assert "expected_operations" not in calls[0]
    assert result["completed_and_verified"] == 1
    assert result["successful_required_operations"] == 1
    assert result["successful_unwanted_writes"] == 0
