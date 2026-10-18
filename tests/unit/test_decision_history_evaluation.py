"""Offline paired-evaluation integrity and reporting checks; no Docker or Bedrock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import GOLDEN_SCENARIO

from agentops_demo.taskify.integrity import harbor_task_sha256, scenario_sha256
from agentops_demo.validation.scenario import load_scenario
from experiments import decision_model
from experiments.decision_model import harbor
from experiments.decision_model.history_evaluate import digest, evaluate, task_order, write
from experiments.decision_model.history_report import (
    analyze,
    markdown,
    operation_diagnostic,
    summarize,
    terminal_invariants,
)


@pytest.fixture
def dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Keep real scenario/tree/freeze hashing; replace only paid execution boundaries."""
    tasks = {}
    payload = load_scenario(GOLDEN_SCENARIO).model_dump(mode="json")
    payload["benchmark"] = {
        "category": "billing",
        "archetype": "test-history",
        "difficulty": "compound",
        "mutation_policy": {"allowed_targets": ["inv-123"]},
        "trajectory": {"required_inspections": ["inv-123"]},
    }
    for index in range(2):
        task = tmp_path / f"task-{index}"
        task.mkdir()
        write(task / "scenario.yaml", payload)
        tasks[task.name] = {
            "task_path": str(task),
            "prototype_id": f"family-{index}",
            "scenario_sha256": scenario_sha256(load_scenario(task / "scenario.yaml")),
            "execution_task_sha256": harbor_task_sha256(task),
            "primary_history_probe": index == 0,
            "unobserved_refund_history_control": index == 1,
        }
    cohort = tmp_path / "cohort.json"
    write(cohort, {"cohort_id": "dry-test", "tasks": tasks})
    policy = Path(harbor.__file__)
    write(
        tmp_path / "policy-freeze.json",
        {
            "cohort_manifest_sha256": digest(cohort),
            "source_sha256": {str(policy): digest(policy)},
        },
    )
    wheel, config = tmp_path / "test.whl", tmp_path / "config.json"
    wheel.write_bytes(b"offline fixture")
    config.write_text("{}")
    state = SimpleNamespace(
        started=[], closed=[], models=[], episodes=[], fail_cleanup=False, fail_start=False
    )

    class FakeHarbor:
        def __init__(self, task, wheel, output, policy_mode, *, observe_escalations):
            self.identity = (task.name, observe_escalations)
            self.task = SimpleNamespace(instruction="inspect inv-123")

        async def start(self):
            state.started.append(self.identity)
            if state.fail_start:
                raise RuntimeError("fixture setup failure")

        async def invoke(self, name, arguments):
            raise AssertionError("fake episode must not call actual tools")

        async def verify(self):
            return {"rewards": {"reward": 1.0}}

        async def close(self):
            state.closed.append(self.identity)
            if state.fail_cleanup:
                raise RuntimeError("fixture cleanup failure")

    def fallback(path):
        instance = object()
        state.models.append(instance)
        return instance

    async def episode(**kwargs):
        state.episodes.append(kwargs)
        assert kwargs["fallback"] is state.models[0]
        assert kwargs["max_tool_calls"] == 16
        if kwargs.get("invoice_authorization"):
            assert kwargs["predict"]({"request": "inspect inv-123"})["action"] == "no_action"
        return {
            "policy_completed": True,
            "llm_calls": 1,
            "llm_input_tokens": 7,
            "llm_output_tokens": 3,
            "tool_call_count": 1,
            "tool_errors": 0,
            "fallback_count": 1,
            "successful_forbidden_writes": 0,
            "tool_observations": [],
            "end_to_end_policy_ms": 2,
            "invoice_authorization": kwargs.get("invoice_authorization", False),
            "authorization_blocked_attempts": 0,
        }

    monkeypatch.setattr(harbor, "NativeHarbor", FakeHarbor)
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", "123456789012")
    monkeypatch.setattr(harbor, "llm_fallback", fallback)
    monkeypatch.setattr(harbor, "run_episode", episode)
    # evaluate deliberately sets these; monkeypatch restores them after the test.
    for name in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "BILLING_MODEL_TEMPERATURE",
        "BILLING_MODEL_MAX_TOKENS",
        "BILLING_MODEL_TOTAL_ATTEMPTS",
    ):
        monkeypatch.setenv(name, "offline-before-test")
    state.args = argparse.Namespace(
        cohort=cohort,
        wheel=wheel,
        fallback_config=config,
        output=tmp_path / "evaluation",
        max_job_seconds=1800,
    )
    return state


@pytest.fixture
def fake_authorization(tmp_path, monkeypatch):
    source = tmp_path / "authorization.py"
    source.write_text("# offline API wiring fixture\n")
    calls = []

    def scoped_rule_action(state, targets, *, escalation_history):
        calls.append((state, targets, escalation_history))
        return "no_action"

    module = SimpleNamespace(
        __file__=str(source),
        VARIANT="invoice-scoped-authorization-v1",
        SERIALIZATION_VERSION="billing-invoice-authorization-v1",
        scoped_rule_action=scoped_rule_action,
    )
    monkeypatch.setattr(decision_model, "authorization", module, raising=False)
    monkeypatch.setitem(sys.modules, "experiments.decision_model.authorization", module)
    return SimpleNamespace(module=module, source=source, calls=calls)


async def test_opt_in_guard_is_same_in_both_arms_and_frozen(dry_run, fake_authorization):
    args = dry_run.args
    args.invoice_authorization = True
    # A follow-up freeze lives separately from the unchanged original cohort/freeze.
    args.policy_freeze = args.cohort.parent / "followup-freeze.json"
    freeze = json.loads((args.cohort.parent / "policy-freeze.json").read_text())
    freeze["source_sha256"][str(fake_authorization.source)] = digest(fake_authorization.source)
    write(args.policy_freeze, freeze)
    await evaluate(args)
    assert [call[2] for call in fake_authorization.calls] == [False, True, True, False]
    assert all(call[1] == ["inv-123"] for call in fake_authorization.calls)
    assert all(episode["invoice_authorization"] for episode in dry_run.episodes)
    combined = json.loads((args.output / "evaluation-manifest.json").read_text())
    assert combined["schema_version"] == "paired-invoice-authorization-history-execution-v1"
    assert combined["invoice_authorization"]
    assert combined["source_sha256"]["authorization.py"] == digest(fake_authorization.source)
    assert (
        args.output / "source-authorization.py"
    ).read_bytes() == fake_authorization.source.read_bytes()
    report = analyze(args.cohort, args.output / "off", args.output / "on")
    assert report["invoice_authorization"]
    assert "previously evaluated families" in markdown(report)
    assert "unchanged invoice-intent parser" not in markdown(report)


async def test_guard_missing_from_freeze_rejected_before_execution(dry_run, fake_authorization):
    dry_run.args.invoice_authorization = True
    with pytest.raises(ValueError, match="authorization source is absent"):
        await evaluate(dry_run.args)
    assert dry_run.started == dry_run.models == []


def test_safety_refusal_is_distinct_from_success_or_unknown_usage():
    report = summarize(
        [
            {
                "task_id": "refused",
                "invoice_authorization": True,
                "authorization_blocked_attempts": 2,
                "policy_completed": False,
                "verifier": {"rewards": {"reward": 1}},
            },
            {
                "task_id": "recovered",
                "invoice_authorization": True,
                "authorization_blocked_attempts": 1,
                "policy_completed": True,
                "verifier": {"rewards": {"reward": 1}},
            },
        ]
    )
    assert report["authorization_blocked_attempts"] == 3
    assert report["tasks_with_authorization_blocks"] == 2
    assert report["completed_after_authorization_block"] == 1
    assert report["unresolved_after_authorization_block"] == 1
    assert report["completed_and_verified"] == 1
    assert summarize([{"task_id": "old"}])["authorization_blocked_attempts"] == 0
    assert (
        summarize([{"task_id": "new", "invoice_authorization": True}])[
            "authorization_blocked_attempts"
        ]
        is None
    )


def test_missing_required_writes_distinct_from_execution_guard_blocks():
    result = operation_diagnostic(
        {
            "withheld": {"tool_observations": []},
            "blocked": {
                "tool_observations": [
                    {
                        "name": "refund_invoice",
                        "arguments": {"invoice_id": "b"},
                        "result": {"ok": False},
                        "authorization_blocked": True,
                    }
                ]
            },
            "complete": {
                "tool_observations": [
                    {
                        "name": "refund_invoice",
                        "arguments": {"invoice_id": "c"},
                        "result": {"ok": True},
                        "authorization_blocked": False,
                    }
                ]
            },
        },
        {
            "withheld": {"expected_operations": {"a": "refund_invoice"}},
            "blocked": {"expected_operations": {"b": "refund_invoice"}},
            "complete": {"expected_operations": {"c": "refund_invoice"}},
        },
    )
    assert result["missing_successful_expected_operations"] == 2
    assert result["tasks_missing_successful_expected_operations"] == 2
    assert result["authorization_blocked_attempts"] == 1
    assert result["successful_expected_operations"] == 1
    assert result["write_attempts_forwarded_past_authorization"] == 1


async def test_alternating_pairing_shared_model_semantic_hashes_and_report(dry_run):
    await evaluate(dry_run.args)
    assert dry_run.started == [
        ("task-0", False),
        ("task-0", True),
        ("task-1", True),
        ("task-1", False),
    ]
    assert dry_run.closed == dry_run.started
    assert len(dry_run.models) == 1
    output = dry_run.args.output
    manifest = json.loads((output / "evaluation-manifest.json").read_text())
    assert manifest["complete"]
    assert manifest["invoice_authorization"] is False
    assert "authorization.py" not in manifest["source_sha256"]
    assert manifest["usage"] == {"llm_calls": 4, "input_tokens": 28, "output_tokens": 12}
    arm = json.loads((output / "off/manifest.json").read_text())
    assert arm["task_scenario_sha256"] != arm["task_scenario_raw_sha256"]
    report = analyze(dry_run.args.cohort, output / "off", output / "on")
    assert report["paired_differences"]["both_completed"] == 2
    assert report["diagnostic_slices"]["primary_history_probe"]["task_count"] == 1
    assert report["diagnostic_slices"]["unobserved_refund_history_control"]["task_count"] == 1
    assert report["confound_checks"]["unrecorded_fields"] == []


@pytest.mark.parametrize("mutation", ["cohort", "source", "scenario", "tree"])
async def test_hash_mismatch_prevents_any_execution(dry_run, mutation):
    args = dry_run.args
    if mutation == "cohort":
        args.cohort.write_text(args.cohort.read_text() + "\n")
    elif mutation == "source":
        freeze = args.cohort.parent / "policy-freeze.json"
        payload = json.loads(freeze.read_text())
        payload["source_sha256"][str(Path(harbor.__file__))] = "0" * 64
        write(freeze, payload)
    else:
        payload = json.loads(args.cohort.read_text())
        field = "scenario_sha256" if mutation == "scenario" else "execution_task_sha256"
        payload["tasks"]["task-0"][field] = "0" * 64
        write(args.cohort, payload)
        freeze = args.cohort.parent / "policy-freeze.json"
        frozen = json.loads(freeze.read_text())
        frozen["cohort_manifest_sha256"] = digest(args.cohort)
        write(freeze, frozen)
    with pytest.raises(ValueError, match=r"differs|changed"):
        await evaluate(args)
    assert dry_run.started == []
    assert dry_run.models == []


async def test_failed_cleanup_stops_before_next_environment(dry_run):
    dry_run.fail_cleanup = True
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await evaluate(dry_run.args)
    assert len(dry_run.started) == len(dry_run.closed) == 1
    manifest = json.loads((dry_run.args.output / "evaluation-manifest.json").read_text())
    assert not manifest["complete"]
    assert manifest["usage"]["llm_calls"] == 1
    assert len(manifest["execution_order"]) == 1


async def test_setup_errors_leave_report_usage_unknown(dry_run):
    dry_run.fail_start = True
    await evaluate(dry_run.args)
    report = analyze(dry_run.args.cohort, dry_run.args.output / "off", dry_run.args.output / "on")
    for arm in report["arms"].values():
        assert arm["execution_errors"] == 2
        assert arm["completed_and_verified"] == 0
        assert arm["llm_calls"] is None
        assert arm["llm_calls_observed_tasks"] == 0
        assert arm["llm_input_tokens"] is None


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "settings", "verifier", "guard"])
async def test_report_rejects_invalid_pairs_or_mismatched_provenance(dry_run, mutation):
    await evaluate(dry_run.args)
    output = dry_run.args.output
    path = output / "on/results.jsonl"
    lines = path.read_text().splitlines()
    if mutation == "missing":
        path.write_text(lines[0] + "\n")
    elif mutation == "duplicate":
        path.write_text(path.read_text() + lines[0] + "\n")
    elif mutation == "verifier":
        row = json.loads(lines[0])
        row["verifier_pass"] = False
        path.write_text(json.dumps(row) + "\n" + lines[1] + "\n")
    else:
        manifest = output / "on/manifest.json"
        payload = json.loads(manifest.read_text())
        if mutation == "guard":
            payload["invoice_authorization"] = True
        else:
            payload["max_tool_calls"] = 17
        write(manifest, payload)
    with pytest.raises(ValueError):
        analyze(dry_run.args.cohort, output / "off", output / "on")


def test_verifier_pass_without_policy_completion_is_not_success():
    value = summarize(
        [
            {
                "task_id": "unfinished",
                "verifier": {"rewards": {"reward": 1.0}},
                "policy_completed": False,
            }
        ]
    )
    assert value["verifier_passes"] == 1
    assert value["completed_and_verified"] == 0
    assert value["llm_calls"] is None


def test_terminal_state_diagnostic_does_not_replace_trajectory_or_completion():
    row = {
        "task_id": "duplicate-rejected",
        "policy_completed": True,
        "verifier": {"rewards": {"reward": 0, "trajectory": 0, "unchanged:invoice": 1}},
    }
    value = summarize([row])
    assert value["terminal_state_invariants_passes"] == 1
    assert value["terminal_state_invariants_observed_tasks"] == 1
    assert value["completed_and_verified"] == 0
    assert terminal_invariants({"verifier": {"rewards": {"reward": 1, "trajectory": 1}}}) is None
    assert terminal_invariants({"verifier": {"rewards": {"must_refund": 0}}}) is False


def test_family_interleaving_is_deterministic():
    assert task_order(
        {
            "b2": {"prototype_id": "b"},
            "a2": {"prototype_id": "a"},
            "a1": {"prototype_id": "a"},
            "b1": {"prototype_id": "b"},
        }
    ) == ["a1", "b1", "a2", "b2"]


def test_posthoc_operation_diagnostic_catches_wrong_action_on_allowed_target():
    rows = {
        "task": {
            "tool_observations": [
                {
                    "name": "get_invoice",
                    "arguments": {"invoice_id": "allowed"},
                    "result": {"ok": True},
                },
                {
                    "name": "refund_invoice",
                    "arguments": {"invoice_id": "allowed"},
                    "result": {"ok": True},
                },
                {
                    "name": "escalate_dispute",
                    "arguments": {"invoice_id": "allowed"},
                    "result": {"ok": True},
                },
                {
                    "name": "escalate_dispute",
                    "arguments": {"invoice_id": "other"},
                    "result": {"ok": False},
                },
            ]
        }
    }
    result = operation_diagnostic(
        rows, {"task": {"expected_operations": {"allowed": "refund_invoice"}}}
    )
    assert result["write_attempts"] == 3
    assert result["successful_writes"] == 2
    assert result["unexpected_operation_attempts"] == 2
    assert result["successful_unexpected_operations"] == 1
    assert operation_diagnostic({"task": {}}, {"task": {}})["write_attempts"] is None
