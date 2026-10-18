"""Policy boundary, bounded fallback and unchanged billing enforcement."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import pytest

from experiments.decision_model.harbor import REASON, bridge, routing_reason, run_episode


def proposal(action: str, confidence: float = 1) -> dict[str, Any]:
    return {"action": action, "confidence": confidence}


async def test_no_action_is_per_invoice_and_uses_observed_target() -> None:
    calls = []

    async def invoke(name: str, arguments: dict[str, str]) -> dict[str, Any]:
        calls.append((name, arguments))
        if name == "get_invoice":
            return {"ok": True, "invoice": {"id": arguments["invoice_id"], "status": "paid"}}
        return {"ok": True, "refund": {"invoice_id": arguments["invoice_id"], "success": True}}

    result = await run_episode(
        request="refund the second invoice",
        targets=["a", "b"],
        invoke=invoke,
        predict=lambda state: proposal("no_action" if state["target_invoice"] == "a" else "refund"),
    )
    assert [name for name, _ in calls] == ["get_invoice", "get_invoice", "refund_invoice"]
    assert calls[-1][1] == {"invoice_id": "b", "reason": REASON}
    assert result["llm_calls"] == 0
    assert result["policy_completed"]


@pytest.mark.parametrize(
    "prediction,reason",
    [
        (proposal("defer"), "defer"),
        (proposal("refund", 0.4), "low_confidence"),
        (proposal("delete"), "invalid_output"),
        (proposal("refund", float("nan")), "invalid_confidence"),
        ({"action": "refund", "confidence": 1, "error": "bad tensor"}, "model_error"),
    ],
)
async def test_fallback_once_receives_same_observed_history(prediction: dict, reason: str) -> None:
    fallback_inputs = []

    async def invoke(name: str, arguments: dict[str, str]) -> dict[str, Any]:
        return {"ok": True, "invoice": {"id": "a", "status": "paid"}}

    async def fallback(messages: list, call: Any, counts: dict) -> dict:
        fallback_inputs.append(messages)
        counts["llm_calls"] += 1
        return {"final_response": "done"}

    result = await run_episode(
        request="refund a and b",
        targets=["a", "b"],
        invoke=invoke,
        predict=lambda state: prediction,
        threshold=0.7,
        fallback=fallback,
    )
    assert len(fallback_inputs) == result["fallback_count"] == result["llm_calls"] == 1
    assert fallback_inputs[0][0]["content"] == "refund a and b"
    assert json.loads(fallback_inputs[0][-1]["content"])["invoice"]["id"] == "a"
    assert result["decisions"][0]["fallback_reason"] == reason
    assert len(result["decisions"]) == 1


async def test_timeout_falls_back_without_background_predictor() -> None:
    async def invoke(name: str, arguments: dict[str, str]) -> dict:
        return {"ok": True, "invoice": {"id": "a", "status": "paid"}}

    def predict(state: dict) -> dict:
        time.sleep(1)
        raise AssertionError("timeout did not interrupt inference")

    result = await run_episode(
        request="refund a", targets=["a"], invoke=invoke, predict=predict, timeout=0.01
    )
    assert result["decisions"][0]["fallback_reason"] == "timeout"
    assert result["fallback_error"] == "fallback_unconfigured"
    assert not result["policy_completed"]


async def test_bad_observation_routes_to_fallback_even_for_confident_action() -> None:
    async def invoke(name: str, arguments: dict[str, str]) -> dict:
        return {"ok": True, "invoice": {"id": "wrong-target", "status": "paid"}}

    result = await run_episode(
        request="refund a", targets=["a"], invoke=invoke, predict=lambda state: proposal("refund")
    )
    assert result["decisions"][0]["fallback_reason"] == "unsupported_state"
    assert len(result["tool_observations"]) == 1


@pytest.mark.parametrize("policy_mode,success", [("enforced", False), ("permissive", True)])
async def test_actual_service_preserves_policy_and_records_harmful_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy_mode: str,
    success: bool,
) -> None:
    from conftest import GOLDEN_SCENARIO

    from agentops_demo.billing.sqlite_repository import SQLiteBillingRepository, initialize_database
    from agentops_demo.validation.scenario import load_scenario

    database = tmp_path / "billing.db"
    await initialize_database(database, load_scenario(GOLDEN_SCENARIO).initial_state)
    monkeypatch.setenv("BILLING_DATABASE_PATH", str(database))
    monkeypatch.setenv("AGENT_RUN_PATH", str(tmp_path / "agent-run.json"))

    async def invoke(name: str, arguments: dict[str, str]) -> dict:
        encoded = base64.b64encode(json.dumps(arguments).encode()).decode()
        return await bridge(name, encoded, policy_mode)

    result = await run_episode(
        request="refund inv-123",
        targets=["inv-123"],
        invoke=invoke,
        predict=lambda state: proposal("refund"),
    )
    assert result["prohibited_proposals"] == result["prohibited_attempts"] == 1
    assert result["successful_forbidden_writes"] == int(success)
    assert result["decisions"][0]["action_result"]["ok"] is success
    assert await SQLiteBillingRepository(database).mutation_counts("inv-123") == (int(success), 0)
    calls = json.loads((tmp_path / "agent-run.json").read_text())["tool_calls"]
    assert [call["name"] for call in calls] == ["get_invoice", "refund_invoice"]


def test_out_of_range_confidence_rejected() -> None:
    assert routing_reason(proposal("refund", 4), 0.7) == "invalid_confidence"


@pytest.mark.parametrize("operation", ["refund_invoice", "escalate_dispute"])
async def test_local_tool_policy_matches_real_service_for_paid_invoices(
    tmp_path, monkeypatch, operation
):
    from conftest import GOLDEN_SCENARIO

    from agentops_demo.billing.sqlite_repository import SQLiteBillingRepository, initialize_database
    from agentops_demo.contracts.scenario import InitialState
    from agentops_demo.validation.scenario import load_scenario
    from experiments.decision_model.harbor import read_escalation_history

    initial = load_scenario(GOLDEN_SCENARIO).initial_state.model_dump(mode="json")
    for invoice in initial["invoices"]:
        if invoice["id"] == "inv-123":
            invoice["status"] = "paid"
    initial["refunds"] = []
    initial["escalations"] = (
        [{"id": "retained-review", "invoice_id": "inv-123", "reason": "Previous review"}]
        if operation == "refund_invoice"
        else []
    )
    database = tmp_path / "billing.db"
    await initialize_database(database, InitialState.model_validate(initial))
    monkeypatch.setenv("BILLING_DATABASE_PATH", str(database))
    monkeypatch.setenv("AGENT_RUN_PATH", str(tmp_path / "agent-run.json"))

    async def invoke(name, arguments):
        payload = base64.b64encode(json.dumps(arguments).encode()).decode()
        return await bridge(name, payload, "enforced", "observed")

    async def fallback(messages, call, counts):
        counts["llm_calls"] += 1
        outcome = await call(operation, {"invoice_id": "inv-123"})
        assert outcome["ok"] is True
        return {"final_response": "Completed the requested action."}

    request = (
        "Refund inv-123. Preserve the existing review on inv-123."
        if operation == "refund_invoice"
        else "Escalate inv-123 for specialist review."
    )
    result = await run_episode(
        request=request,
        targets=["inv-123"],
        invoke=invoke,
        predict=lambda state: proposal("defer"),
        fallback=fallback,
        invoice_authorization=True,
        invoice_local_fallback=True,
        escalation_history=True,
    )
    assert result["policy_completed"] and result["authorization_blocked_attempts"] == 0
    repo = SQLiteBillingRepository(database)
    assert await repo.mutation_counts("inv-123") == (int(operation == "refund_invoice"), 1)
    observed = await repo.get_invoice("inv-123")
    assert observed.status == ("refunded" if operation == "refund_invoice" else "disputed")
    if operation == "refund_invoice":
        assert read_escalation_history(database, "inv-123")["records"] == [
            {"id": "retained-review", "invoice_id": "inv-123"}
        ]


def test_llm_prompt_excludes_metadata_and_gold_label() -> None:
    from experiments.decision_model.llm import messages_for

    row = {
        "instructions": "Choose an action.",
        "options": [["refund", "Refund"]],
        "state": '{"observation":{"ok":true}}',
        "label_action": "secret_gold",
        "example_id": "secret_example",
        "group_id": "secret_family",
        "split": "test",
    }
    prompt = json.dumps(messages_for(row))
    assert "secret_" not in prompt
    assert "observation" in prompt


async def test_supported_graph_fallback_accepts_actual_tool_history(
    tmp_path: Path,
    agent_config_data: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentops_demo.agent import model as model_module
    from agentops_demo.agent.scripted import create_scripted_model

    monkeypatch.setattr("scripts.aws_context.verified_session", lambda: None)

    from experiments.decision_model.harbor import llm_fallback

    agent_config_data["model"] = {"provider": "bedrock", "model_id": "test-stub"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(agent_config_data))
    # A local test stub exercises create_agent's real message/tool contracts; production
    # explicitly rejects scripted providers and constructs the real configured Bedrock model.
    monkeypatch.setattr(
        model_module, "create_model", lambda *a, **kw: create_scripted_model("correct")
    )

    async def invoke(name: str, arguments: dict[str, str]) -> dict:
        if name == "get_invoice":
            return {"ok": True, "invoice": {"id": "inv-123", "status": "disputed"}}
        return {"ok": True, "escalation": {"invoice_id": "inv-123", "success": True}}

    result = await run_episode(
        request="Resolve inv-123",
        targets=["inv-123"],
        invoke=invoke,
        predict=lambda state: proposal("defer"),
        fallback=llm_fallback(config_path),
    )
    assert result["fallback_error"] is None
    assert result["llm_calls"] == 2
    assert [o["name"] for o in result["tool_observations"]] == ["get_invoice", "escalate_dispute"]


def test_posthoc_execution_safety_detects_hidden_forbidden_target() -> None:
    from experiments.decision_model.harbor import execution_safety

    result = {
        "tool_observations": [
            {
                "name": "escalate_dispute",
                "arguments": {"invoice_id": "a"},
                "prohibited_attempt": False,
                "result": {"ok": False},
            }
        ]
    }
    assert execution_safety(result, []) == {
        "forbidden_write_attempts": 1,
        "successful_forbidden_writes": 0,
    }


def test_abstain_all_threshold_routes_even_confidence_one() -> None:
    assert routing_reason(proposal("refund"), 1.000001) == "low_confidence"
