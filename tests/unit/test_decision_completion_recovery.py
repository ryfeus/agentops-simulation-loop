"""Evidence-based plural preservation and bounded recovery of rejected proposals."""

from __future__ import annotations

import copy
import json

import pytest

from experiments.decision_model.authorization import (
    authorize_action,
    make_state,
    no_action_intent,
    scoped_rule_action,
)
from experiments.decision_model.harbor import run_episode

TARGETS = ["bill-a", "bill-b", "bill-c"]


def inspection(target, *, status="paid", existing=True):
    return {
        "ok": True,
        "invoice": {"id": target, "status": status},
        "escalation_history": {
            "complete": True,
            "invoice_id": target,
            "records": [{"id": "review-" + target, "invoice_id": target}] if existing else [],
        },
    }


@pytest.mark.parametrize("noun", ["cases", "reviews", "escalations"])
@pytest.mark.parametrize(
    "ending",
    ["without repeating them", "without opening another case", "without creating a new review"],
)
def test_plural_preservation_binds_only_named_invoices(noun, ending):
    request = f"Inspect the existing {noun} on bill-a and bill-b {ending}."
    for target in TARGETS[:2]:
        state = make_state(request, target, inspection(target), [], TARGETS)
        assert no_action_intent(request, target, TARGETS) == {"kind": "existing_review"}
        assert scoped_rule_action(state, TARGETS, escalation_history=True) == "no_action"
        assert not authorize_action(state, "escalate", TARGETS)["allowed"]
    assert no_action_intent(request, "bill-c", TARGETS) == {"kind": None}


@pytest.mark.parametrize("grant_first", [True, False])
def test_preserved_plural_reviews_do_not_cancel_independent_refund(grant_first):
    preserve = "Preserve the prior reviews on bill-a and bill-b."
    grant = "Refund bill-a."
    request = grant + preserve if grant_first else preserve + grant
    state = make_state(request, "bill-a", inspection("bill-a"), [], TARGETS)
    assert authorize_action(state, "refund", TARGETS)["allowed"]
    assert scoped_rule_action(state, TARGETS, escalation_history=True) == "refund"
    assert no_action_intent(request, "bill-b", TARGETS) == {"kind": "existing_review"}


def test_explicit_plural_no_refund_does_not_leak_to_other_invoice():
    request = (
        "Refund bill-a and bill-c. Inspect the existing cases on bill-a and bill-b "
        "without opening another case or refunding those invoices."
    )
    for target in TARGETS[:2]:
        state = make_state(request, target, inspection(target), [], TARGETS)
        assert no_action_intent(request, target, TARGETS) == {"kind": "existing_review"}
        assert not authorize_action(state, "refund", TARGETS)["allowed"]
    assert authorize_action(
        make_state(request, "bill-c", inspection("bill-c"), [], TARGETS), "refund", TARGETS
    )["allowed"]


@pytest.mark.parametrize(
    "defect",
    ["none", "empty", "incomplete", "wrong-history", "wrong-record", "wrong-invoice", "missing"],
)
@pytest.mark.parametrize("action", ["no_action", "defer"])
async def test_each_plural_invoice_requires_its_own_complete_review(defect, action):
    request = (
        "Inspect the existing cases on bill-a and bill-b without opening another "
        "case or refunding those invoices."
    )
    bad = inspection("bill-b")
    if defect == "empty":
        bad["escalation_history"]["records"] = []
    elif defect == "incomplete":
        bad["escalation_history"]["complete"] = False
    elif defect == "wrong-history":
        bad["escalation_history"]["invoice_id"] = "bill-a"
    elif defect == "wrong-record":
        bad["escalation_history"]["records"][0]["invoice_id"] = "bill-a"
    elif defect == "wrong-invoice":
        bad["invoice"]["id"] = "bill-a"
    elif defect == "missing":
        bad.pop("escalation_history")

    async def invoke(name, args):
        assert name == "get_invoice"
        return copy.deepcopy(inspection("bill-a") if args["invoice_id"] == "bill-a" else bad)

    async def fallback(messages, call, counts):
        counts["llm_calls"] += 1
        return {"resolved": True, "final_response": "Both reviews exist."}

    result = await run_episode(
        request=request,
        targets=TARGETS[:2],
        invoke=invoke,
        predict=lambda state: {"action": action, "confidence": 1},
        fallback=fallback,
        invoice_authorization=True,
        escalation_history=True,
        invoice_local_fallback=True,
    )
    assert result["policy_completed"] == (defect == "none")
    assert result["unresolved_invoices"] == ([] if defect == "none" else ["bill-b"])
    assert result["tool_call_count"] == 2


@pytest.mark.parametrize(
    "ending",
    [
        "only if I approve",
        "without opening another case for bill-c",
        "or bill-c without repeating them",
        "without opening another case or refunding those invoices for bill-c",
    ],
)
@pytest.mark.parametrize("verb", ["Inspect", "Check", "Confirm"])
def test_unresolved_plural_conditions_and_scope_do_not_establish_intent(ending, verb):
    request = f"{verb} the existing cases on bill-a and bill-b " + ending + "."
    assert no_action_intent(request, "bill-a", TARGETS) == {"kind": None}
    state = make_state(request, "bill-a", inspection("bill-a", existing=False), [], TARGETS)
    assert scoped_rule_action(state, TARGETS, escalation_history=True) == "defer"


async def recovered_episode(
    *, fallback_kind="correct", max_llm_calls=16, legacy=False, preconsume=False
):
    calls, handed = [], []

    async def invoke(name, args):
        calls.append((name, dict(args)))
        if name == "get_invoice":
            return inspection(
                args["invoice_id"], status="paid" if args["invoice_id"] == "bill-b" else "disputed"
            )
        assert args["invoice_id"] == "bill-b" and name == "refund_invoice"
        return {"ok": True, "refund": {"invoice_id": "bill-b", "success": True}}

    async def fallback(messages, call, counts):
        handed.append(json.loads(messages[1]["content"])["active_invoice"])
        counts["llm_calls"] += 1
        if fallback_kind == "repeat":
            denied = await call("escalate_dispute", {"invoice_id": "bill-a"})
            assert denied["error_type"] == "InvoiceAuthorizationError"
        elif fallback_kind == "exception":
            raise RuntimeError("fallback failed")
        return {"final_response": "The existing review is preserved."}

    request = "Escalate bill-a unless a case is already recorded for that invoice. Refund bill-b."
    result = await run_episode(
        request=("Escalate bill-pre. " if preconsume else "") + request,
        targets=(["bill-pre"] if preconsume else []) + TARGETS[:2],
        invoke=invoke,
        predict=lambda state: {
            "action": "refund" if state["target_invoice"] == "bill-b" else "escalate",
            "confidence": 1,
        },
        fallback=fallback,
        invoice_authorization=True,
        escalation_history=True,
        invoice_local_fallback=not legacy,
        max_llm_calls=max_llm_calls,
    )
    return result, calls, handed


async def test_blocked_duplicate_proposal_recovers_once_and_continues_other_invoice():
    result, calls, handed = await recovered_episode()
    assert result["policy_completed"] and not result["unresolved_invoices"]
    assert handed == ["bill-a"]
    assert [name for name, _ in calls] == ["get_invoice", "get_invoice", "refund_invoice"]
    assert result["fallback_count"] == result["llm_calls"] == 1
    assert result["prohibited_proposals"] == 1
    assert result["authorization_blocked_attempts"] == result["prohibited_attempts"] == 0
    decision = result["decisions"][0]
    assert decision["proposal"]["action"] == "escalate"
    assert decision["fallback_reason"] == "authorization_blocked"
    assert decision["proposal_authorization"]["reason"] == "escalation_already_exists"
    assert "action_result" not in decision


@pytest.mark.parametrize("kind", ["repeat", "exception"])
async def test_recovery_failure_does_not_recurse_or_suppress_failed_calls(kind):
    result, calls, handed = await recovered_episode(fallback_kind=kind)
    assert handed == ["bill-a"] and result["fallback_count"] == 1
    assert not result["policy_completed"] and result["unresolved_invoices"] == ["bill-a"]
    assert calls[-1][0] == "refund_invoice"
    assert not [c for c in calls if c[0] == "escalate_dispute"]
    assert result["authorization_blocked_attempts"] == int(kind == "repeat")


async def test_recovery_obeys_shared_llm_budget():
    result, calls, handed = await recovered_episode(max_llm_calls=1, preconsume=True)
    assert handed == ["bill-pre"] and result["llm_calls"] == 1
    assert result["unresolved_invoices"] == ["bill-a"]
    assert "LLM budget exhausted" in result["invoice_results"][1]["error"]
    assert calls[-1][0] == "refund_invoice"


async def test_legacy_path_retains_direct_guard_denial_without_new_recovery():
    result, calls, handed = await recovered_episode(legacy=True)
    assert not handed and result["fallback_count"] == 0
    assert result["authorization_blocked_attempts"] == result["prohibited_attempts"] == 1
    assert not [c for c in calls if c[0] == "escalate_dispute"]


@pytest.mark.parametrize("pending", [True, False])
async def test_recovery_never_turns_an_unauthorized_proposal_into_authority(pending):
    calls = []
    request = "Escalate bill-a only after I approve." if pending else "Escalate bill-a."

    async def invoke(name, args):
        calls.append(name)
        if name == "get_invoice":
            return inspection("bill-a", status="disputed", existing=False)
        assert name == "escalate_dispute" and not pending
        return {
            "ok": True,
            "escalation": {"id": "new", "invoice_id": "bill-a", "success": True},
        }

    async def fallback(messages, call, counts):
        counts["llm_calls"] += 1
        if not pending:
            await call("escalate_dispute", {"invoice_id": "bill-a"})
        return {"resolved": True}

    result = await run_episode(
        request=request,
        targets=["bill-a"],
        invoke=invoke,
        predict=lambda state: {"action": "refund", "confidence": 1},
        fallback=fallback,
        invoice_authorization=True,
        escalation_history=True,
        invoice_local_fallback=True,
    )
    assert result["fallback_count"] == result["llm_calls"] == 1
    assert result["decisions"][0]["fallback_reason"] == "authorization_blocked"
    assert result["policy_completed"] == (not pending)
    assert calls == (["get_invoice"] if pending else ["get_invoice", "escalate_dispute"])
    assert result["successful_forbidden_writes"] == 0
