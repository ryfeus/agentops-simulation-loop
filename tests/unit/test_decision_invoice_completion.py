"""Completion requires both request intent and target-matched observed evidence."""

from __future__ import annotations

import copy

import pytest

from experiments.decision_model.harbor import run_episode


def review_observation():
    return {
        "ok": True,
        "invoice": {"id": "inv-1", "status": "paid"},
        "escalation_history": {
            "complete": True,
            "invoice_id": "inv-1",
            "records": [{"id": "existing-review", "invoice_id": "inv-1"}],
        },
    }


async def episode(request, observation, *, action="defer", fallback_result=None):
    calls = []

    async def invoke(name, arguments):
        calls.append((name, dict(arguments)))
        assert name == "get_invoice", "A completion claim must not perform writes"
        return copy.deepcopy(observation)

    async def fallback(messages, call, counts):
        counts["llm_calls"] += 1
        return fallback_result or {"final_response": "Reviewed the available observations."}

    result = await run_episode(
        request=request,
        targets=["inv-1"],
        invoke=invoke,
        predict=lambda state: {"action": action, "confidence": 1},
        fallback=fallback,
        invoice_authorization=True,
        escalation_history=True,
        invoice_local_fallback=True,
    )
    assert calls == [("get_invoice", {"invoice_id": "inv-1"})]
    return result


@pytest.mark.parametrize("action", ["defer", "no_action"])
@pytest.mark.parametrize(
    "request_text",
    [
        "For inv-1, inspect the existing review without repeating it or issuing a refund.",
        "Inspect the existing case on inv-1 without repeating it.",
        "Preserve the existing review on inv-1.",
    ],
)
async def test_existing_review_completes_from_matching_observed_history(request_text, action):
    result = await episode(request_text, review_observation(), action=action)
    assert result["policy_completed"]
    assert result["unresolved_invoices"] == []
    assert result["tool_call_count"] == 1
    assert result["fallback_count"] == int(action == "defer")


@pytest.mark.parametrize("action", ["defer", "no_action"])
@pytest.mark.parametrize(
    "defect",
    [
        "missing_history",
        "empty_history",
        "incomplete_history",
        "wrong_history_target",
        "wrong_record_target",
        "missing_record_id",
        "wrong_invoice",
        "missing_invoice",
        "refunded_without_history",
    ],
)
async def test_review_preservation_never_invents_missing_or_mismatched_evidence(defect, action):
    observed = review_observation()
    if defect == "missing_history":
        observed.pop("escalation_history")
    elif defect == "empty_history":
        observed["escalation_history"]["records"] = []
    elif defect == "incomplete_history":
        observed["escalation_history"]["complete"] = False
    elif defect == "wrong_history_target":
        observed["escalation_history"]["invoice_id"] = "inv-other"
    elif defect == "wrong_record_target":
        observed["escalation_history"]["records"][0]["invoice_id"] = "inv-other"
    elif defect == "missing_record_id":
        observed["escalation_history"]["records"][0].pop("id")
    elif defect == "wrong_invoice":
        observed["invoice"]["id"] = "inv-other"
    elif defect == "missing_invoice":
        observed = {"ok": False, "error_type": "InvoiceNotFoundError"}
    elif defect == "refunded_without_history":
        observed["invoice"]["status"] = "refunded"
        observed.pop("escalation_history")
    result = await episode(
        "Inspect the existing case on inv-1 without repeating it.",
        observed,
        action=action,
        fallback_result={"resolved": True, "final_response": "I confirmed the existing review."},
    )
    assert result["policy_completed"] is False
    assert result["unresolved_invoices"] == ["inv-1"]
    assert result["successful_forbidden_writes"] == 0


@pytest.mark.parametrize("action", ["defer", "no_action"])
@pytest.mark.parametrize(
    "request_text",
    [
        "Refund inv-1. Preserve the existing review on inv-1.",
        "Refund inv-1 only after I approve. Preserve the existing review on inv-1.",
        "Please process inv-1.",
    ],
)
async def test_existing_history_does_not_complete_missing_writes_or_unknown_intent(
    request_text, action
):
    result = await episode(request_text, review_observation(), action=action)
    assert result["policy_completed"] is False
    assert result["unresolved_invoices"] == ["inv-1"]


async def test_status_only_request_does_not_require_existing_review_history():
    observed = review_observation()
    observed.pop("escalation_history")
    result = await episode("Report the status of inv-1 only.", observed)
    assert result["policy_completed"] and result["unresolved_invoices"] == []


async def test_completed_review_inspection_does_not_drop_later_authorized_refund():
    calls = []

    async def invoke(name, arguments):
        calls.append((name, dict(arguments)))
        if arguments["invoice_id"] == "inv-1":
            assert name == "get_invoice"
            return review_observation()
        if name == "get_invoice":
            return {"ok": True, "invoice": {"id": "inv-2", "status": "paid"}}
        assert name == "refund_invoice"
        return {"ok": True, "refund": {"invoice_id": "inv-2", "success": True}}

    result = await run_episode(
        request="For inv-1, inspect the existing review without repeating it. Refund inv-2.",
        targets=["inv-1", "inv-2"],
        invoke=invoke,
        predict=lambda state: {
            "action": "no_action" if state["target_invoice"] == "inv-1" else "refund",
            "confidence": 1,
        },
        invoice_authorization=True,
        escalation_history=True,
        invoice_local_fallback=True,
    )
    assert result["policy_completed"]
    assert [name for name, _ in calls] == ["get_invoice", "get_invoice", "refund_invoice"]
    assert result["unresolved_invoices"] == []
