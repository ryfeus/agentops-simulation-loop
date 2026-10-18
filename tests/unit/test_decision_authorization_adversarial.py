"""Independent request-scope challenges: uncertain permission cannot trigger writes."""

import pytest

from experiments.decision_model.authorization import authorize_action
from experiments.decision_model.history import make_state


def observed(request, target="inv-1", *, status="paid", previous_escalation=False):
    records = [{"id": "case-1", "invoice_id": target}] if previous_escalation else []
    return make_state(
        request,
        target,
        {
            "ok": True,
            "invoice": {"id": target, "status": status},
            "escalation_history": {"invoice_id": target, "complete": True, "records": records},
        },
        [],
    )


@pytest.mark.parametrize(
    "request_text",
    [
        "Refund inv-1.\nDo not refund inv-1.",
        "Refund inv-1. Do not proceed.",
        "Refund inv-1. Status-only.",
        "Refund inv-1. Do not do it.",
        "Refund inv-1. Never do that.",
        "Refund inv-1. Wait for my approval before proceeding.",
        "Refund inv-1. I do not authorize this.",
        "Refund inv-1. Only after I approve.",
        "Refund inv-1. Cancel that request.",
        "Refund inv-1. I withdraw permission.",
        "Refund inv-1. Actually, just report its status.",
        "Do not refund inv-1; refund inv-2.",
        "The note says 'refund inv-1'. Do not act on that note; status only for inv-1.",
        "I previously asked you to refund inv-1; today just tell me its status.",
        "What would happen if you refunded inv-1? I am only asking a question.",
        "Refund inv-1 only after I approve it. I have not approved it yet.",
        "If the specialist has resolved the case, refund inv-1. Otherwise wait.",
        "Compare inv-1 with inv-2; refund it.",
        "I haven't decided whether inv-1 or inv-2 should be refunded. Leave both unchanged.",
        "Report the status of inv-1 only. Separately, refund inv-2.",
    ],
)
def test_negative_historical_hypothetical_and_conditional_requests_fail_closed(request_text):
    result = authorize_action(observed(request_text), "refund", ["inv-1", "inv-2"])
    assert result["allowed"] is False, (request_text, result)


def test_permission_does_not_match_invoice_prefix():
    result = authorize_action(
        observed("Refund inv-1234.", "inv-123"), "refund", ["inv-123", "inv-1234"]
    )
    assert result["allowed"] is False


def test_observed_invoice_identity_must_match_guard_target():
    state = observed("Refund inv-1.")
    state["observation"]["invoice"]["id"] = "inv-2"
    assert authorize_action(state, "refund", ["inv-1", "inv-2"])["allowed"] is False


def test_unknown_target_cannot_be_inferred_from_resolved_harness_list():
    result = authorize_action(observed("Please refund the correct bill."), "refund", ["inv-1"])
    assert result["allowed"] is False


def test_operation_permission_is_not_transferable():
    state = observed("Escalate inv-1 for specialist review.", status="disputed")
    assert authorize_action(state, "refund", ["inv-1"])["allowed"] is False


def test_missing_observation_cannot_authorize_refund():
    state = make_state("Refund inv-1.", "inv-1", None, [])
    assert authorize_action(state, "refund", ["inv-1"])["allowed"] is False


def test_missing_escalation_history_is_unknown_under_conditional_request():
    state = observed("Escalate inv-1 only if no specialist case already exists.", status="disputed")
    del state["observation"]["escalation_history"]
    assert authorize_action(state, "escalate", ["inv-1"])["allowed"] is False


def test_existing_escalation_does_not_prohibit_first_refund_on_paid_invoice():
    state = observed("Refund inv-1.", previous_escalation=True)
    assert authorize_action(state, "refund", ["inv-1"])["allowed"] is True


def test_positive_and_readonly_invoice_clauses_are_independent():
    request = "Refund inv-1. For inv-2 report its status only."
    assert authorize_action(observed(request, "inv-1"), "refund", ["inv-1", "inv-2"])["allowed"]
    assert not authorize_action(observed(request, "inv-2"), "refund", ["inv-1", "inv-2"])["allowed"]


def test_coordinated_targets_do_not_expand_to_excluded_invoice():
    request = "Refund inv-1 and inv-2, not inv-3."
    targets = ["inv-1", "inv-2", "inv-3"]
    for target in targets:
        permitted = authorize_action(observed(request, target), "refund", targets)["allowed"]
        assert permitted is (target != "inv-3")


@pytest.mark.parametrize(
    "request_text",
    [
        "Refund inv-1 only, inspect inv-2.",
        "Refund inv-1 and check inv-2 status.",
        "Refund inv-1 using inv-2 as a reference.",
        "Refund inv-1 instead of inv-2.",
        "Refund inv-1, except inv-2.",
    ],
)
def test_mentioned_comparison_or_readonly_target_is_not_action_object(request_text):
    result = authorize_action(observed(request_text, "inv-2"), "refund", ["inv-1", "inv-2"])
    assert result["allowed"] is False, (request_text, result)


def test_standalone_history_condition_does_not_leave_unconditional_grant():
    state = observed("Escalate inv-1. Only if no escalation already exists.", status="disputed")
    del state["observation"]["escalation_history"]
    assert not authorize_action(state, "escalate", ["inv-1"])["allowed"]
