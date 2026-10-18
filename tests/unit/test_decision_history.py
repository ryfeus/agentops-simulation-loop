"""Live history is observed data, never hidden scenario metadata or assumed absence."""

from __future__ import annotations

import base64
import json
import sqlite3

import pytest

from experiments.decision_model import data, history
from experiments.decision_model.harbor import bridge, read_escalation_history, run_episode


def encoded(**kwargs):
    return base64.b64encode(json.dumps(kwargs).encode()).decode()


@pytest.fixture
async def database(tmp_path, monkeypatch):
    from conftest import GOLDEN_SCENARIO

    from agentops_demo.billing.sqlite_repository import initialize_database
    from agentops_demo.contracts.scenario import InitialState
    from agentops_demo.validation.scenario import load_scenario

    initial = load_scenario(GOLDEN_SCENARIO).initial_state.model_dump(mode="json")
    initial["invoices"].append({**initial["invoices"][0], "id": "other"})
    initial["escalations"] = [{"id": "existing-other", "invoice_id": "other", "reason": "private"}]
    path = tmp_path / "billing.db"
    await initialize_database(path, InitialState.model_validate(initial))
    monkeypatch.setenv("BILLING_DATABASE_PATH", str(path))
    monkeypatch.setenv("AGENT_RUN_PATH", str(tmp_path / "agent-run.json"))
    return path


async def test_opt_in_read_is_target_scoped_and_reflects_live_writes(database):
    payload = encoded(invoice_id="inv-123")
    original = await bridge("get_invoice", payload, "enforced")
    observed = await bridge("get_invoice", payload, "enforced", "observed")
    assert original == {
        key: value for key, value in observed.items() if key != "escalation_history"
    }
    assert observed["escalation_history"] == {
        "invoice_id": "inv-123",
        "complete": True,
        "records": [],
    }
    await bridge("escalate_dispute", encoded(invoice_id="inv-123", reason="request"), "enforced")
    latest = await bridge("get_invoice", payload, "enforced", "observed")
    assert latest["escalation_history"]["records"] == [
        {"id": "escalation-inv-123", "invoice_id": "inv-123"}
    ]
    assert "private" not in json.dumps(latest) and "existing-other" not in json.dumps(latest)
    assert read_escalation_history(database, "inv-123' OR 1=1 --")["records"] == []


async def test_duplicate_is_avoided_but_service_guard_still_blocks_it(database):
    from agentops_demo.billing.sqlite_repository import SQLiteBillingRepository

    await bridge("escalate_dispute", encoded(invoice_id="inv-123", reason="earlier"), "enforced")

    async def invoke(name, arguments):
        return await bridge(name, encoded(**arguments), "enforced", "observed")

    result = await run_episode(
        request="Please escalate inv-123",
        targets=["inv-123"],
        invoke=invoke,
        predict=lambda state: {"action": history.rule_action(state), "confidence": 1},
        escalation_history=True,
    )
    assert result["policy_completed"] and result["fallback_count"] == 0
    assert [o["name"] for o in result["tool_observations"]] == ["get_invoice"]
    bad = await run_episode(
        request="Please escalate inv-123",
        targets=["inv-123"],
        invoke=invoke,
        predict=lambda state: {"action": "escalate", "confidence": 1},
        escalation_history=True,
    )
    assert bad["prohibited_proposals"] == bad["prohibited_attempts"] == 1
    assert bad["successful_forbidden_writes"] == 0
    assert await SQLiteBillingRepository(database).mutation_counts("inv-123") == (0, 1)


async def test_unavailable_history_is_not_empty_and_missing_invoice_unchanged(database):
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE escalations")
    obs = await bridge("get_invoice", encoded(invoice_id="inv-123"), "enforced", "observed")
    assert obs["ok"] and obs["escalation_history"]["complete"] is False
    assert "records" not in obs["escalation_history"]
    assert history.rule_action(history.make_state("Resolve inv-123", "inv-123", obs, [])) == "defer"
    missing = await bridge("get_invoice", encoded(invoice_id="absent"), "enforced", "observed")
    assert missing["error_type"] == "InvoiceNotFoundError"
    assert (
        history.rule_action(history.make_state("Refund absent", "absent", missing, []))
        == "no_action"
    )


@pytest.mark.parametrize(
    "bad_history",
    [
        None,
        {},
        {"complete": True, "invoice_id": "other", "records": []},
        {"complete": True, "invoice_id": "a", "records": [{"id": "x", "invoice_id": "other"}]},
    ],
)
def test_unknown_or_cross_target_history_requires_defer(bad_history):
    obs = {
        "ok": True,
        "invoice": {"id": "a", "status": "disputed"},
        "escalation_history": bad_history,
    }
    state = history.make_state("Resolve a", "a", obs, [])
    assert history.rule_action(state) == "defer"


def test_escalation_history_does_not_prohibit_first_paid_refund():
    obs = {
        "ok": True,
        "invoice": {"id": "a", "status": "paid"},
        "escalation_history": {
            "complete": True,
            "invoice_id": "a",
            "records": [{"id": "esc", "invoice_id": "a"}],
        },
    }
    state = history.make_state("Refund a", "a", obs, [])
    assert history.rule_action(state) == "refund"
    assert history.prohibited_actions(state) == ["escalate"]


async def test_fallback_and_rule_share_exact_enriched_observation(database):
    seen = []

    async def invoke(name, arguments):
        return await bridge(name, encoded(**arguments), "enforced", "observed")

    def predict(state):
        seen.append(state)
        return {"action": "defer", "confidence": 1}

    async def fallback(messages, call, counts):
        assert json.loads(messages[-1]["content"]) == seen[0]["observation"]
        assert await call("get_invoice", {"invoice_id": "inv-123"}) == seen[0]["observation"]
        counts["llm_calls"] += 1
        return {"final_response": "observed"}

    result = await run_episode(
        request="Check inv-123 history",
        targets=["inv-123"],
        invoke=invoke,
        predict=predict,
        fallback=fallback,
        escalation_history=True,
    )
    assert result["fallback_count"] == result["llm_calls"] == 1
    assert seen[0]["prior_actions"] == []
    assert set(seen[0]) == set(data.make_state("", "", None, []))


@pytest.mark.parametrize(
    "request_text", ["Do not escalate invoice a.", "Please escalate invoice a."]
)
def test_history_does_not_reinterpret_ambiguous_intent_parser(request_text):
    obs = {
        "ok": True,
        "invoice": {"id": "a", "status": "disputed"},
        "escalation_history": {"complete": True, "invoice_id": "a", "records": []},
    }
    state = history.make_state(request_text, "a", obs, [])
    assert history.rule_action(state) == data.rule_action(state) == "defer"
