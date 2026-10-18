"""Authorization v2 semantics and real SQLite/native verifier regression cohort."""

import base64
import importlib.util
import json
from pathlib import Path

import pytest

from experiments.decision_model.authorization import (
    authorization,
    authorize_action,
    make_state,
    scoped_rule_action,
)


def observed(request, target="inv-1", status="paid", records=()):
    return make_state(
        request,
        target,
        {
            "ok": True,
            "invoice": {"id": target, "status": status},
            "escalation_history": {
                "invoice_id": target,
                "complete": True,
                "records": list(records),
            },
        },
        [],
        ["inv-1", "inv-2", "inv-3"],
    )


@pytest.mark.parametrize(
    "suffix",
    [
        "Their matching amounts do not make them interchangeable.",
        "Do not postpone that independent review because my refund choice is unresolved.",
    ],
)
def test_benign_negative_predicates_do_not_cancel_authority(suffix):
    request = "Refund inv-1. Escalate inv-2. " + suffix
    assert authorize_action(observed(request), "refund", ["inv-1", "inv-2"])["allowed"]
    assert authorize_action(observed(request, "inv-2", "disputed"), "escalate", ["inv-1", "inv-2"])[
        "allowed"
    ]


def test_typed_escalation_condition_keeps_independent_refund():
    request = "For customer-7, refund inv-1 and escalate inv-2 only if no case already exists."
    assert authorize_action(observed(request), "refund", ["inv-1", "inv-2"])["allowed"]
    review = observed(request, "inv-2", "disputed")
    assert authorize_action(review, "escalate", ["inv-1", "inv-2"])["allowed"]
    review["observation"].pop("escalation_history")
    assert not authorize_action(review, "escalate", ["inv-1", "inv-2"])["allowed"]


def test_repeat_constraint_preserves_predicate_and_missing_history_is_unknown():
    request = "Escalate inv-1. Never refund a disputed charge or repeat an existing escalation."
    sample = observed(request, status="disputed")
    assert authorize_action(sample, "escalate", ["inv-1"])["allowed"]
    sample["observation"].pop("escalation_history")
    assert scoped_rule_action(sample, ["inv-1"], escalation_history=True) == "defer"


def test_prior_refund_condition_requires_unavailable_ledger():
    sample = observed("Refund inv-1. Do not repeat an existing refund.")
    assert scoped_rule_action(sample, ["inv-1"]) == "defer"


def test_uncertainty_differs_from_explicit_no_action():
    sample = observed("Refund inv-1 only after I approve.")
    assert sample["authorization"]["status"] == "ambiguous"
    assert scoped_rule_action(sample, ["inv-1"]) == "defer"
    sample = observed("Leave inv-1 unchanged until I approve.")
    assert scoped_rule_action(sample, ["inv-1"]) == "no_action"


def test_unscoped_stop_revokes_all_previous_commands_and_features_are_not_trusted():
    request = "Refund inv-1. Refund inv-2. Do not proceed."
    assert not authorization(request, "inv-1", ["inv-1", "inv-2"])["allowed_actions"]
    sample = observed("Leave inv-1 unchanged.")
    sample["authorization"] = {"allowed_actions": ["refund"], "status": "authorized"}
    assert not authorize_action(sample, "refund", ["inv-1"])["allowed"]


@pytest.mark.parametrize(
    "suffix",
    [
        "For inv-1, do not refund it.",
        "Do not repeat it.",
    ],
)
def test_unknown_or_scoped_negative_never_disappears(suffix):
    request = "Refund inv-1. " + suffix
    assert not authorize_action(observed(request), "refund", ["inv-1"])["allowed"]


def test_unknown_mixed_condition_cannot_be_split_away():
    request = "Refund inv-1 and escalate inv-2 only after I approve."
    for target, action in [("inv-1", "refund"), ("inv-2", "escalate")]:
        assert not authorize_action(observed(request, target), action, ["inv-1", "inv-2"])[
            "allowed"
        ]


async def test_known_history_cohort_actual_sqlite_and_native_verifier(tmp_path, monkeypatch):
    from agentops_demo.billing.sqlite_repository import initialize_database
    from experiments.decision_model.harbor import bridge, run_episode
    from experiments.decision_model.history_scenarios import scenarios

    verifier_path = (
        Path(__file__).resolve().parents[2] / "benchmarks/disputed-refund/tests/verify.py"
    )
    spec = importlib.util.spec_from_file_location("auth_v2_native_verifier", verifier_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    outcomes = []
    writes = 0
    for name, scenario in scenarios().items():
        directory = tmp_path / name
        directory.mkdir()
        database = directory / "billing.db"
        trajectory = directory / "agent-run.json"
        source = directory / "scenario.json"
        source.write_text(scenario.model_dump_json())
        await initialize_database(database, scenario.initial_state)
        monkeypatch.setenv("BILLING_DATABASE_PATH", str(database))
        monkeypatch.setenv("AGENT_RUN_PATH", str(trajectory))
        targets = scenario.benchmark.trajectory.required_inspections
        calls = []

        async def invoke(name, args, calls=calls):
            calls.append(name)
            payload = base64.b64encode(json.dumps(args).encode()).decode()
            return await bridge(name, payload, "enforced", "observed")

        result = await run_episode(
            request=scenario.instruction,
            targets=targets,
            invoke=invoke,
            predict=lambda state, targets=targets: {
                "action": scoped_rule_action(state, targets, escalation_history=True),
                "confidence": 1,
            },
            escalation_history=True,
            invoice_authorization=True,
        )
        verifier.CONTAINER_SCENARIO_PATH = source
        verifier.AGENT_RESULT_PATH = trajectory
        rewards, diagnostics = verifier.evaluate_database(database)
        writes += sum(name != "get_invoice" for name in calls)
        outcomes.append((name, rewards["reward"], result["policy_completed"], diagnostics))
        assert result["successful_forbidden_writes"] == 0
    assert [
        (name, reward, complete)
        for name, reward, complete, _ in outcomes
        if reward != 1 or not complete
    ] == [], outcomes
    assert len(outcomes) == 24
    assert writes == 30
