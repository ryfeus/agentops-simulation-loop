"""Opt-in invoice-local continuation boundaries with shared budgets and real scope guard."""

from __future__ import annotations

import asyncio
import json

import pytest

from experiments.decision_model.harbor import run_episode


class Billing:
    def __init__(self, statuses=None):
        self.statuses = statuses or {"inv-1": "paid", "inv-2": "paid", "inv-3": "paid"}
        self.calls = []

    async def invoke(self, name, args):
        self.calls.append((name, dict(args)))
        target = args["invoice_id"]
        if name == "get_invoice":
            if target not in self.statuses:
                return {"ok": False, "error_type": "InvoiceNotFoundError"}
            return {
                "ok": True,
                "invoice": {"id": target, "status": self.statuses[target]},
                "escalation_history": {"complete": True, "invoice_id": target, "records": []},
            }
        if name == "refund_invoice":
            self.statuses[target] = "refunded"
            return {"ok": True, "refund": {"invoice_id": target, "success": True}}
        return {"ok": True, "escalation": {"invoice_id": target, "success": True}}


def proposal(action):
    return {"action": action, "confidence": 1}


async def run(billing, *, request="Refund inv-1. Refund inv-2.", targets=None, **kwargs):
    return await run_episode(
        request=request,
        targets=targets or ["inv-1", "inv-2"],
        invoke=billing.invoke,
        invoice_authorization=True,
        escalation_history=True,
        invoice_local_fallback=True,
        **kwargs,
    )


async def test_unresolved_local_invoice_does_not_drop_independent_later_write():
    billing = Billing()
    contexts = []
    request = "Refund inv-1 only after I approve. Separately, refund inv-2."

    async def fallback(messages, call, counts):
        counts["llm_calls"] += 1
        contexts.append(messages)
        return {"final_response": "Please confirm inv-1", "resolved": True}

    result = await run(
        billing,
        request=request,
        predict=lambda s: proposal("defer" if s["target_invoice"] == "inv-1" else "refund"),
        fallback=fallback,
    )
    assert billing.statuses["inv-2"] == "refunded"
    assert result["policy_completed"] is False
    assert result["unresolved_invoices"] == ["inv-1"]
    assert result["invoice_results"][0]["status"] == "unresolved"
    assert result["fallback_count"] == result["llm_calls"] == 1
    assert contexts[0][0]["role"] == "system"
    assert contexts[0][1]["role"] == "user"
    assignment = json.loads(contexts[0][1]["content"])
    assert assignment["original_request_reference"] == request
    assert assignment["active_invoice"] == "inv-1"
    assert "Reuse the existing inspection" in contexts[0][0]["content"]
    assert [
        json.loads(m["content"])["invoice"]["id"] for m in contexts[0] if m["role"] == "tool"
    ] == ["inv-1"]


async def test_fallback_sees_only_target_history_and_rejects_cross_invoice_reads_and_writes():
    billing = Billing()
    received = []

    async def fallback(messages, call, counts):
        counts["llm_calls"] += 1
        received.extend(messages)
        for name in ("get_invoice", "refund_invoice"):
            denied = await call(name, {"invoice_id": "inv-1", "reason": "cross target"})
            assert denied["error_type"] == "InvoiceScopeError"
        assert (await call("refund_invoice", {"invoice_id": "inv-2", "reason": "current target"}))[
            "ok"
        ]
        return {}

    result = await run(
        billing,
        predict=lambda s: proposal("refund" if s["target_invoice"] == "inv-1" else "defer"),
        fallback=fallback,
    )
    tool_results = [json.loads(m["content"]) for m in received if m["role"] == "tool"]
    assert len(tool_results) == 1 and tool_results[0]["invoice"]["id"] == "inv-2"
    assert [args["invoice_id"] for name, args in billing.calls if name == "refund_invoice"] == [
        "inv-1",
        "inv-2",
    ]
    assert result["invoice_scope_blocked_attempts"] == 2
    assert result["tool_call_count"] == 6  # Includes both rejected attempts.
    assert result["unresolved_invoices"] == [
        "inv-2"
    ]  # Errors remain visible despite later success.


async def test_successful_target_fallback_resolves_and_keeps_shared_counts():
    billing = Billing()

    async def fallback(messages, call, counts):
        counts["llm_calls"] += 1
        target = json.loads(next(m["content"] for m in messages if m["role"] == "tool"))["invoice"][
            "id"
        ]
        await call("refund_invoice", {"invoice_id": target, "reason": "approved"})
        return {"final_response": "Done"}

    result = await run(billing, predict=lambda s: proposal("defer"), fallback=fallback)
    assert result["policy_completed"]
    assert result["llm_calls"] == result["fallback_count"] == 2
    assert result["tool_call_count"] == 4
    assert result["unresolved_invoices"] == []


async def test_local_continuation_cannot_reset_shared_tool_budget_or_repeat_forever():
    billing = Billing()

    async def fallback(messages, call, counts):
        counts["llm_calls"] += 1
        while True:
            await call("get_invoice", {"invoice_id": "inv-3"})

    result = await run(
        billing, predict=lambda s: proposal("defer"), fallback=fallback, max_tool_calls=3
    )
    assert result["tool_call_count"] == 3
    assert len(billing.calls) == 1  # Cross-target attempts never reach tools.
    assert result["invoice_scope_blocked_attempts"] == 2
    assert result["llm_calls"] == 1
    assert result["unresolved_invoices"] == ["inv-1", "inv-2"]
    assert all(r["status"] == "error" for r in result["invoice_results"])


async def test_shared_llm_budget_prevents_later_continuation_but_not_independent_rules():
    billing = Billing()
    called = []

    async def fallback(messages, call, counts):
        counts["llm_calls"] += 1
        called.append(1)
        await call("refund_invoice", {"invoice_id": "inv-1", "reason": "authorized"})
        return {}

    result = await run(
        billing,
        request="Refund inv-1 and inv-2 and inv-3.",
        targets=["inv-1", "inv-2", "inv-3"],
        predict=lambda s: proposal("refund" if s["target_invoice"] == "inv-3" else "defer"),
        fallback=fallback,
        max_llm_calls=1,
    )
    assert called == [1] and result["llm_calls"] == 1
    assert billing.statuses["inv-3"] == "refunded"
    assert result["unresolved_invoices"] == ["inv-2"]
    assert "LLM budget exhausted" in result["invoice_results"][1]["error"]


@pytest.mark.parametrize("error", ["exception", "timeout"])
async def test_local_fallback_errors_terminate_and_continue(error):
    billing = Billing()

    async def fallback(messages, call, counts):
        if error == "exception":
            raise ValueError("failure")
        await asyncio.sleep(1)

    result = await run(
        billing,
        predict=lambda s: proposal("defer" if s["target_invoice"] == "inv-1" else "refund"),
        fallback=fallback,
        fallback_timeout=0.01,
    )
    assert billing.statuses["inv-2"] == "refunded"
    assert result["unresolved_invoices"] == ["inv-1"]
    assert result["invoice_results"][0]["status"] == "error"


async def test_scope_authority_overrides_legacy_global_readonly_flag_but_enforces_denials():
    billing = Billing()
    request = "Report the status of inv-1 only. Refund inv-2."

    async def fallback(messages, call, counts):
        counts["llm_calls"] += 1
        target = json.loads(next(m["content"] for m in messages if m["role"] == "tool"))["invoice"][
            "id"
        ]
        result = await call("refund_invoice", {"invoice_id": target, "reason": "same proposal"})
        assert result["ok"] is (target == "inv-2")
        return {}

    result = await run(
        billing, request=request, predict=lambda s: proposal("defer"), fallback=fallback
    )
    assert result["prohibited_attempts"] == result["authorization_blocked_attempts"] == 1
    assert result["successful_forbidden_writes"] == 0
    assert billing.statuses["inv-1"] == "paid" and billing.statuses["inv-2"] == "refunded"


@pytest.mark.parametrize("status", ["disputed", "open", "refunded"])
async def test_scoped_refund_still_enforces_observed_billing_status(status):
    billing = Billing({"inv-1": status})
    result = await run(
        billing, targets=["inv-1"], request="Refund inv-1.", predict=lambda s: proposal("refund")
    )
    assert not [c for c in billing.calls if c[0] != "get_invoice"]
    assert result["prohibited_proposals"] == 1
    assert result["successful_forbidden_writes"] == 0


@pytest.mark.parametrize("target", ["inv-1", "inv-missing"])
async def test_readonly_and_missing_fallback_can_finish_from_observed_facts(target):
    billing = Billing()

    async def fallback(messages, call, counts):
        return {"final_response": "Status reported"}

    result = await run(
        billing,
        targets=[target],
        request=f"Report the status of {target} only.",
        predict=lambda s: proposal("defer"),
        fallback=fallback,
    )
    assert result["policy_completed"] and not result["unresolved_invoices"]


async def test_legacy_default_still_stops_after_one_whole_task_handoff():
    billing = Billing()
    result = await run_episode(
        request="Refund inv-1. Refund inv-2.",
        targets=["inv-1", "inv-2"],
        invoke=billing.invoke,
        predict=lambda s: proposal("defer"),
        invoice_authorization=True,
    )
    assert result["invoice_local_fallback"] is False
    assert result["fallback_count"] == 1 and result["tool_call_count"] == 1
    assert not result["policy_completed"]


async def test_inspection_using_last_tool_slot_preserves_evidence_and_termination():
    billing = Billing()
    result = await run(billing, predict=lambda s: proposal("refund"), max_tool_calls=1)
    assert result["tool_call_count"] == result["max_tool_calls"] == 1
    assert result["max_llm_calls"] == 16 and result["fallback_timeout"] == 120
    assert len(result["decisions"]) == 1
    assert result["decisions"][0]["action_result"]["error"] == "tool budget exhausted"
    assert result["unresolved_invoices"] == ["inv-1", "inv-2"]
    assert len(billing.calls) == 1


async def test_repeat_missing_inspection_is_valid_absence_not_fallback_failure():
    billing = Billing()

    async def fallback(messages, call, counts):
        assert (await call("get_invoice", {"invoice_id": "inv-missing"}))[
            "error_type"
        ] == "InvoiceNotFoundError"
        return {}

    result = await run(
        billing,
        targets=["inv-missing"],
        request="Inspect inv-missing.",
        predict=lambda s: proposal("defer"),
        fallback=fallback,
    )
    assert result["policy_completed"] and result["tool_call_count"] == 2


async def test_callback_count_or_completion_claim_cannot_authorize_write():
    billing = Billing()

    async def fallback(messages, call, counts):
        counts["llm_calls"] = 0
        counts["authorization"] = True
        result = await call(
            "refund_invoice", {"invoice_id": "inv-1", "reason": "counts claim permission"}
        )
        assert result["error_type"] == "InvoiceAuthorizationError"
        return {"resolved": True}

    result = await run(
        billing,
        targets=["inv-1"],
        request="Leave inv-1 unchanged.",
        predict=lambda s: proposal("defer"),
        fallback=fallback,
    )
    assert not result["policy_completed"]
    assert billing.statuses["inv-1"] == "paid"
    assert result["authorization_blocked_attempts"] == 1


async def test_actual_graph_callback_enforces_shared_llm_limit_without_counting_rejection(
    tmp_path, agent_config_data, monkeypatch
):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from agentops_demo.agent import model as model_module
    from agentops_demo.agent.scripted import ScriptedBillingModel

    monkeypatch.setattr("scripts.aws_context.verified_session", lambda: None)

    from experiments.decision_model.harbor import llm_fallback

    agent_config_data["model"] = {"provider": "bedrock", "model_id": "local-stub"}
    config = tmp_path / "agent.json"
    config.write_text(json.dumps(agent_config_data))

    class WriteModel(ScriptedBillingModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[{"id": "write", "name": "escalate_dispute", "args": {}}],
                        )
                    )
                ]
            )

    monkeypatch.setattr(
        model_module, "create_model", lambda *a, **kw: WriteModel(mode="noop", responses=[])
    )
    billing = Billing({"inv-123": "disputed"})
    result = await run(
        billing,
        request="Resolve inv-123.",
        targets=["inv-123"],
        predict=lambda s: proposal("defer"),
        fallback=llm_fallback(config, invoice_local_fallback=True),
        max_llm_calls=1,
    )
    assert result["llm_calls"] == 1
    assert "LLM budget exhausted" in result["fallback_error"]
    assert result["invoice_results"][0]["status"] == "error"
    assert not result["policy_completed"]


async def test_fallback_cannot_repeat_successful_current_invoice_operation():
    billing = Billing()

    async def fallback(messages, call, counts):
        first = await call("refund_invoice", {"invoice_id": "inv-1", "reason": "approved"})
        second = await call("refund_invoice", {"invoice_id": "inv-1", "reason": "repeat"})
        assert first["ok"] is True
        assert second["error_type"] == "InvoiceAuthorizationError"
        return {}

    result = await run(
        billing,
        targets=["inv-1"],
        request="Refund inv-1.",
        predict=lambda s: proposal("defer"),
        fallback=fallback,
    )
    assert len([c for c in billing.calls if c[0] == "refund_invoice"]) == 1
    assert result["prohibited_attempts"] == result["authorization_blocked_attempts"] == 1
    assert not result["policy_completed"]


async def test_scoped_proposal_risk_keeps_existing_escalation_prohibition():
    billing = Billing({"inv-1": "disputed"})

    async def invoke(name, args):
        result = await billing.invoke(name, args)
        if name == "get_invoice":
            result["escalation_history"]["records"] = [{"id": "previous", "invoice_id": "inv-1"}]
        return result

    result = await run_episode(
        request="Escalate inv-1.",
        targets=["inv-1"],
        invoke=invoke,
        predict=lambda s: proposal("escalate"),
        invoice_authorization=True,
        escalation_history=True,
        invoice_local_fallback=True,
    )
    assert result["prohibited_proposals"] == 1
    assert result["prohibited_attempts"] == 0
    assert result["decisions"][0]["fallback_reason"] == "authorization_blocked"
    assert result["decisions"][0]["proposal_authorization"]["reason"] == "escalation_already_exists"
    assert not result["policy_completed"]
    assert "fallback_unconfigured" in result["fallback_error"]
    assert not [call for call in billing.calls if call[0] != "get_invoice"]


async def test_task_cancellation_preserves_prior_write_counters_and_remaining_invoices():
    billing = Billing()
    sink = {}
    entered = asyncio.Event()

    async def fallback(messages, call, counts):
        counts["llm_calls"] += 1
        counts["llm_input_tokens"] += 17
        counts["llm_output_tokens"] += 3
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(
        run(
            billing,
            request="Refund inv-1 and inv-2 and inv-3.",
            targets=["inv-1", "inv-2", "inv-3"],
            evidence_sink=sink,
            predict=lambda s: proposal("refund" if s["target_invoice"] == "inv-1" else "defer"),
            fallback=fallback,
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sink["episode_cancelled"] is True and sink["policy_completed"] is False
    assert (
        sink["llm_calls"] == 1 and sink["llm_input_tokens"] == 17 and sink["llm_output_tokens"] == 3
    )
    assert sink["tool_call_count"] == 3 and sink["tool_evidence_complete"] is True
    assert sink["unresolved_invoices"] == ["inv-2", "inv-3"]
    assert sink["invoice_results"][0]["status"] == "resolved"
    assert any(
        o["name"] == "refund_invoice" and o["result"]["ok"] for o in sink["tool_observations"]
    )
    assert billing.statuses["inv-1"] == "refunded"


async def test_cancellation_during_committed_write_marks_unknown_tool_outcome():
    billing = Billing()
    sink, committed = {}, asyncio.Event()

    async def invoke(name, args):
        result = await billing.invoke(name, args)
        if name == "refund_invoice":
            committed.set()
            await asyncio.Event().wait()
        return result

    task = asyncio.create_task(
        run_episode(
            request="Refund inv-1.",
            targets=["inv-1"],
            invoke=invoke,
            predict=lambda s: proposal("refund"),
            invoice_authorization=True,
            invoice_local_fallback=True,
            evidence_sink=sink,
        )
    )
    await committed.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert billing.statuses["inv-1"] == "refunded"
    assert sink["tool_call_count"] == 2
    assert sink["tool_evidence_complete"] is False
    assert sink["in_flight_tool_call"]["name"] == "refund_invoice"
    assert sink["indeterminate_tool_calls"][0]["outcome"] == "unknown_after_cancellation"
    assert sink["policy_completed"] is False


@pytest.mark.parametrize("request_text", ["Refund inv-1.", "Refund inv-1 only after I approve."])
async def test_confident_no_action_needs_observed_scope_justification(request_text):
    billing = Billing()
    result = await run(
        billing, targets=["inv-1"], request=request_text, predict=lambda s: proposal("no_action")
    )
    assert result["decisions"][0]["proposal"]["action"] == "no_action"
    assert result["unresolved_invoices"] == ["inv-1"]
    assert result["invoice_results"][0]["status"] == "unresolved"
    assert result["policy_completed"] is False
    assert result["fallback_count"] == 0 and len(billing.calls) == 1


async def test_local_timeout_keeps_unknown_write_and_consumed_budget_while_continuing():
    billing = Billing()

    async def invoke(name, args):
        result = await billing.invoke(name, args)
        if name == "refund_invoice" and args["invoice_id"] == "inv-1":
            await asyncio.Event().wait()
        return result

    async def fallback(messages, call, counts):
        await call("refund_invoice", {"invoice_id": "inv-1", "reason": "approved"})

    result = await run_episode(
        request="Refund inv-1. Refund inv-2.",
        targets=["inv-1", "inv-2"],
        invoke=invoke,
        predict=lambda s: proposal("defer" if s["target_invoice"] == "inv-1" else "refund"),
        fallback=fallback,
        fallback_timeout=0.01,
        max_tool_calls=4,
        invoice_authorization=True,
        invoice_local_fallback=True,
    )
    assert billing.statuses["inv-1"] == billing.statuses["inv-2"] == "refunded"
    assert result["tool_call_count"] == 4 and len(result["tool_observations"]) == 3
    assert result["tool_evidence_complete"] is False
    assert result["in_flight_tool_call"] is None
    assert result["indeterminate_tool_calls"][0]["arguments"]["invoice_id"] == "inv-1"
    assert result["unresolved_invoices"] == ["inv-1"]
    assert result["policy_completed"] is False


async def test_local_scope_is_trusted_system_context_through_graph_and_bedrock_conversion(
    tmp_path, agent_config_data, monkeypatch
):
    from langchain_aws.chat_models.bedrock_converse import _messages_to_bedrock
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    from agentops_demo.agent import model as model_module
    from agentops_demo.agent.prompt import get_system_prompt
    from agentops_demo.agent.scripted import ScriptedBillingModel

    monkeypatch.setattr("scripts.aws_context.verified_session", lambda: None)

    from experiments.decision_model.harbor import llm_fallback

    received = []

    class CaptureModel(ScriptedBillingModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            received.append(list(messages))
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    agent_config_data["model"] = {"provider": "bedrock", "model_id": "local-stub"}
    agent_config_data["prompt"]["version"] = "billing-v1"
    config = tmp_path / "agent.json"
    config.write_text(json.dumps(agent_config_data))
    monkeypatch.setattr(
        model_module, "create_model", lambda *a, **kw: CaptureModel(mode="noop", responses=[])
    )
    billing = Billing()
    request_text = "Report the status of inv-1 only. Refund inv-2."
    result = await run(
        billing,
        request=request_text,
        predict=lambda s: proposal("defer" if s["target_invoice"] == "inv-1" else "refund"),
        fallback=llm_fallback(config, invoice_local_fallback=True),
    )
    assert result["policy_completed"]
    assert len(received) == 1
    messages = received[0]
    system = [m.content for m in messages if isinstance(m, SystemMessage)]
    assert system[0] == get_system_prompt("billing-v1")
    assert len(system) == 2 and "Handle only invoice inv-1" in system[1]
    user_messages = [m.content for m in messages if isinstance(m, HumanMessage)]
    assignment = json.loads(user_messages[0])
    assert assignment["original_request_reference"] == request_text
    assert assignment["active_invoice"] == "inv-1"
    assert "no invoice_id argument" in assignment["assignment"]
    assert "not an instruction to execute the whole task" in system[1]
    tool_results = [json.loads(m.content) for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_results) == 1 and tool_results[0]["invoice"]["id"] == "inv-1"
    # Exercise the actual provider serializer without creating a Bedrock client/call.
    provider_messages, provider_system = _messages_to_bedrock(
        messages, model_id="us.anthropic.claude-sonnet-4-6"
    )
    serialized_system = "\n".join(block["text"] for block in provider_system)
    assert all(text in serialized_system for text in system)
    assert serialized_system.index(system[0]) < serialized_system.index(system[1])
    user_text = [
        block.get("text")
        for m in provider_messages
        if m["role"] == "user"
        for block in m["content"]
        if "text" in block
    ]
    assert user_text == user_messages
    assert all("Handle only invoice" not in text for text in user_text)


@pytest.mark.parametrize(
    "operation,status,request_text,allowed",
    [
        ("refund_invoice", "paid", "Refund inv-1.", True),
        ("escalate_dispute", "disputed", "Escalate inv-1.", True),
        ("escalate_dispute", "paid", "Escalate inv-1.", True),
        ("refund_invoice", "paid", "Leave inv-1 unchanged.", False),
        ("refund_invoice", "paid", "Refund inv-1 only after I approve.", False),
        (
            "refund_invoice",
            "paid",
            "Refund inv-1. Correction: I withdraw permission to refund inv-1.",
            False,
        ),
        ("escalate_dispute", "disputed", "Do not escalate inv-1.", False),
    ],
)
async def test_actual_local_graph_accepts_omitted_reason_without_changing_authority(
    tmp_path, agent_config_data, monkeypatch, operation, status, request_text, allowed
):
    from langchain_aws.chat_models.bedrock_converse import _format_tools, _messages_to_bedrock
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from agentops_demo.agent import model as model_module
    from agentops_demo.agent.scripted import ScriptedBillingModel

    monkeypatch.setattr("scripts.aws_context.verified_session", lambda: None)

    from experiments.decision_model.harbor import REASON, llm_fallback

    received, bound_tools = [], []

    class OmittedReasonModel(ScriptedBillingModel):
        def bind_tools(self, tools, **kwargs):
            bound_tools.extend(tools)
            return super().bind_tools(tools, **kwargs)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            received.append(list(messages))
            done = any(isinstance(m, ToolMessage) and m.name == operation for m in messages)
            message = (
                AIMessage(content="Finished.")
                if done
                else AIMessage(
                    content="",
                    tool_calls=[{"id": "without-reason", "name": operation, "args": {}}],
                )
            )
            return ChatResult(generations=[ChatGeneration(message=message)])

    agent_config_data["model"] = {"provider": "bedrock", "model_id": "local-stub"}
    agent_config_data["prompt"]["version"] = "billing-v1"
    config = tmp_path / "agent.json"
    config.write_text(json.dumps(agent_config_data))
    monkeypatch.setattr(
        model_module, "create_model", lambda *a, **kw: OmittedReasonModel(mode="noop", responses=[])
    )
    billing = Billing({"inv-1": status})
    result = await run(
        billing,
        request=request_text,
        targets=["inv-1"],
        predict=lambda s: {
            "action": "refund" if operation == "refund_invoice" else "escalate",
            "confidence": 0.1,
        },
        threshold=0.9,
        fallback=llm_fallback(config, invoice_local_fallback=True),
    )
    assert result["decisions"][0]["fallback_reason"] == "low_confidence"
    assert result["llm_calls"] == 2 and result["fallback_count"] == 1
    assert result["policy_completed"] is allowed
    assert result["authorization_blocked_attempts"] == int(not allowed)
    assert result["successful_forbidden_writes"] == 0
    writes = [(name, args) for name, args in billing.calls if name != "get_invoice"]
    assert writes == ([(operation, {"invoice_id": "inv-1", "reason": REASON})] if allowed else [])

    provider_tools = _format_tools(bound_tools)
    write_spec = next(t["toolSpec"] for t in provider_tools if t["toolSpec"]["name"] == operation)
    schema = write_spec["inputSchema"]["json"]
    assert not schema.get("required")
    assert "invoice_id" not in schema["properties"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["reason"]["default"] == REASON
    assert "do not ask the customer for a separate reason" in write_spec["description"]
    if operation == "refund_invoice":
        assert "Do not require a refundable flag" in write_spec["description"]
        assert "existing escalation review does not prevent" in write_spec["description"]
    else:
        assert "both paid and disputed invoices may be escalated" in write_spec["description"]
        assert "need not already have a dispute flag" in write_spec["description"]
    inspection_spec = next(
        t["toolSpec"] for t in provider_tools if t["toolSpec"]["name"] == "get_invoice"
    )
    assert "not that a case has been resolved" in inspection_spec["description"]
    assert inspection_spec["inputSchema"]["json"]["properties"] == {}
    assert inspection_spec["inputSchema"]["json"]["additionalProperties"] is False
    _, provider_system = _messages_to_bedrock(
        received[0], model_id="us.anthropic.claude-sonnet-4-6"
    )
    system_text = "\n".join(block["text"] for block in provider_system)
    assert "neither grants nor revokes authorization" in system_text
    assert "decision model's uncertainty, not customer permission" in system_text
    assert "A separate customer-provided reason is not required" in system_text
    # Raw route labels remain diagnostic telemetry, not evidence in model context.
    assert "low_confidence" not in system_text


async def test_legacy_fallback_keeps_required_reason_schema(
    tmp_path, agent_config_data, monkeypatch
):
    from langchain_aws.chat_models.bedrock_converse import _format_tools

    from agentops_demo.agent import model as model_module
    from agentops_demo.agent.scripted import ScriptedBillingModel

    monkeypatch.setattr("scripts.aws_context.verified_session", lambda: None)

    from experiments.decision_model.harbor import llm_fallback

    bound_tools = []

    class CaptureToolsModel(ScriptedBillingModel):
        def bind_tools(self, tools, **kwargs):
            bound_tools.extend(tools)
            return super().bind_tools(tools, **kwargs)

    agent_config_data["model"] = {"provider": "bedrock", "model_id": "local-stub"}
    config = tmp_path / "agent.json"
    config.write_text(json.dumps(agent_config_data))
    monkeypatch.setattr(
        model_module, "create_model", lambda *a, **kw: CaptureToolsModel(mode="noop", responses=[])
    )
    result = await run_episode(
        request="Refund inv-1.",
        targets=["inv-1"],
        invoke=Billing().invoke,
        predict=lambda s: proposal("defer"),
        fallback=llm_fallback(config),
    )
    assert result["invoice_local_fallback"] is False
    for tool in _format_tools(bound_tools):
        spec = tool["toolSpec"]
        assert (
            spec["description"]
            == {
                "get_invoice": "Inspect an invoice",
                "refund_invoice": "Refund a refundable invoice",
                "escalate_dispute": "Escalate a disputed invoice",
            }[spec["name"]]
        )
        if spec["name"] == "get_invoice":
            continue
        assert set(spec["inputSchema"]["json"]["required"]) == {"invoice_id", "reason"}
        assert "default" not in spec["inputSchema"]["json"]["properties"]["reason"]


@pytest.mark.parametrize(
    "operation,arguments,error_type",
    [
        ("get_invoice", {"invoice_id": "inv-2"}, "InvoiceScopeError"),
        ("refund_invoice", {"invoice_id": "inv-2"}, "InvoiceScopeError"),
        ("refund_invoice", {"invoice_id": "inv-1"}, "InvoiceToolArgumentsError"),
        ("refund_invoice", {"reason": 123}, "InvoiceToolArgumentsError"),
        ("get_invoice", {"target": "inv-2"}, "InvoiceToolArgumentsError"),
    ],
)
async def test_actual_bound_tools_reject_arguments_without_coercion_or_hidden_errors(
    tmp_path, agent_config_data, monkeypatch, operation, arguments, error_type
):
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from agentops_demo.agent import model as model_module
    from agentops_demo.agent.scripted import ScriptedBillingModel

    monkeypatch.setattr("scripts.aws_context.verified_session", lambda: None)

    from experiments.decision_model.harbor import llm_fallback

    observed_errors = []

    class WrongArgumentsModel(ScriptedBillingModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            errors = [
                json.loads(m.content)
                for m in messages
                if isinstance(m, ToolMessage) and m.tool_call_id == "bad-arguments"
            ]
            observed_errors.extend(errors)
            response = (
                AIMessage(content="Claiming success cannot erase the tool error.")
                if errors
                else AIMessage(
                    content="",
                    tool_calls=[{"id": "bad-arguments", "name": operation, "args": arguments}],
                )
            )
            return ChatResult(generations=[ChatGeneration(message=response)])

    agent_config_data["model"] = {"provider": "bedrock", "model_id": "local-stub"}
    config = tmp_path / "agent.json"
    config.write_text(json.dumps(agent_config_data))
    monkeypatch.setattr(
        model_module,
        "create_model",
        lambda *a, **kw: WrongArgumentsModel(mode="noop", responses=[]),
    )
    billing = Billing()
    result = await run(
        billing,
        predict=lambda s: proposal("defer" if s["target_invoice"] == "inv-1" else "refund"),
        fallback=llm_fallback(config, invoice_local_fallback=True),
        max_tool_calls=4,
    )
    assert observed_errors[0]["error_type"] == error_type
    assert result["tool_call_count"] == 4  # Two inspections, rejection, independent refund.
    assert result["tool_errors"] == 1
    assert result["llm_calls"] == 2
    assert result["invoice_results"][0]["error"] == "fallback_tool_error"
    assert result["unresolved_invoices"] == ["inv-1"]
    assert not result["policy_completed"]
    assert billing.statuses["inv-1"] == "paid"  # Never redirect a bad call onto the bound target.
    assert billing.statuses["inv-2"] == "refunded"  # The independent invoice still progresses.
    assert len(billing.calls) == 3  # Rejected call never reaches the billing service.


@pytest.mark.parametrize(
    "request_text,completed",
    [
        (
            "Inspect inv-1 and inv-2. Do not issue another refund for inv-1. "
            "If the refund ledger is unavailable, explain that uncertainty and leave "
            "inv-1 unchanged. Refund inv-2.",
            True,
        ),
        (
            "I have not chosen whether inv-1 should receive a refund; leave inv-1 unchanged. "
            "Independently, refund inv-2.",
            True,
        ),
        ("Refund inv-1 only after I approve. Independently, refund inv-2.", False),
    ],
)
async def test_actual_local_no_action_retains_reference_conditions_and_independent_work(
    tmp_path, agent_config_data, monkeypatch, request_text, completed
):
    from langchain_core.messages import HumanMessage

    from agentops_demo.agent import model as model_module
    from agentops_demo.agent.scripted import ScriptedBillingModel

    monkeypatch.setattr("scripts.aws_context.verified_session", lambda: None)

    from experiments.decision_model.harbor import llm_fallback

    assignments = []

    class LocalAnswerModel(ScriptedBillingModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            assignments.extend(
                json.loads(m.content) for m in messages if isinstance(m, HumanMessage)
            )
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    agent_config_data["model"] = {"provider": "bedrock", "model_id": "local-stub"}
    config = tmp_path / "agent.json"
    config.write_text(json.dumps(agent_config_data))
    monkeypatch.setattr(
        model_module, "create_model", lambda *a, **kw: LocalAnswerModel(mode="noop", responses=[])
    )
    billing = Billing()
    result = await run(
        billing,
        request=request_text,
        predict=lambda s: proposal("defer" if s["target_invoice"] == "inv-1" else "refund"),
        fallback=llm_fallback(config, invoice_local_fallback=True),
    )
    assert assignments[0]["active_invoice"] == "inv-1"
    assert assignments[0]["original_request_reference"] == request_text
    assert "inv-2" not in assignments[0]["assignment"]
    assert all(d["state"]["request"] == request_text for d in result["decisions"])
    assert result["policy_completed"] is completed
    assert result["tool_call_count"] == 3
    assert result["invoice_scope_blocked_attempts"] == 0
    assert billing.statuses["inv-1"] == "paid" and billing.statuses["inv-2"] == "refunded"
