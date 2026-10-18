"""Request scope is independent of scenario expectations and invoice iteration order."""

import pytest

from experiments.decision_model.authorization import (
    authorization,
    authorize_action,
    scoped_rule_action,
)
from experiments.decision_model.history import make_state


def state(request, target="inv-1", status="paid", records=()):
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
    )


def test_actions_attach_to_named_invoices_not_global_keywords():
    request = "Refund inv-1 and escalate inv-2. For inv-3, report its status only."
    targets = ["inv-3", "inv-2", "inv-1"]
    assert authorization(request, "inv-1", targets)["allowed_actions"] == ["refund"]
    assert authorization(request, "inv-2", targets)["allowed_actions"] == ["escalate"]
    assert authorization(request, "inv-3", targets)["status"] == "denied"
    assert not authorize_action(state(request, "inv-2"), "refund", targets)["allowed"]


def test_two_named_refunds_and_different_readonly_target():
    request = "Refund inv-1 and inv-2. Leave inv-3 unchanged."
    for target in ["inv-1", "inv-2"]:
        assert authorize_action(state(request, target), "refund", ["inv-3", "inv-2", "inv-1"])[
            "allowed"
        ]
    assert scoped_rule_action(state(request, "inv-3"), ["inv-1", "inv-2", "inv-3"]) == "no_action"


@pytest.mark.parametrize(
    "request_text",
    [
        "Do not refund inv-1.",
        "Never refund inv-1.",
        "Refund inv-1 only after I approve it.",
        "If I confirm tomorrow, refund inv-1.",
        "I have not decided whether inv-1 or inv-2 should be refunded.",
        'Yesterday I said "refund inv-1"; now report its status only.',
        "What would happen if you refund inv-1?",
        "Consider a refund for inv-1.",
    ],
)
def test_negated_conditional_narrated_and_ambiguous_requests_fail_closed(request_text):
    assert not authorize_action(state(request_text), "refund", ["inv-1", "inv-2"])["allowed"]


def test_later_negation_overrides_grant():
    request = "Refund inv-1. Do not refund inv-1."
    assert not authorize_action(state(request), "refund", ["inv-1"])["allowed"]


@pytest.mark.parametrize(
    "request_text",
    [
        "Do not refund inv-1 and escalate inv-2.",
        "If I approve, refund inv-1 and escalate inv-2.",
        "Refund inv-1 and escalate inv-2 only after I approve.",
    ],
)
def test_conjunction_does_not_strip_shared_negation_or_condition(request_text):
    targets = ["inv-1", "inv-2"]
    assert not authorize_action(state(request_text), "refund", targets)["allowed"]
    assert not authorize_action(state(request_text, "inv-2", "disputed"), "escalate", targets)[
        "allowed"
    ]


def test_unknown_invoice_and_prefix_collision_never_substitute():
    assert (
        authorization("Refund inv-1234", "inv-123", ["inv-123", "inv-1234"])["allowed_actions"]
        == []
    )
    assert authorization("Refund inv-9", "inv-1", ["inv-1"])["allowed_actions"] == []
    assert authorization("Refund inv-9", "inv-9", ["inv-1"])["allowed_actions"] == []


def test_unknown_pronoun_with_two_antecedents_does_not_authorize():
    request = "Inspect inv-1 and inv-2. Refund it."
    assert not authorization(request, "inv-1", ["inv-1", "inv-2"])["allowed_actions"]
    assert not authorization(request, "inv-2", ["inv-1", "inv-2"])["allowed_actions"]


def test_history_condition_requires_complete_actual_observation():
    request = "Escalate inv-1 unless already escalated."
    observed = state(request, status="disputed")
    assert authorize_action(observed, "escalate", ["inv-1"])["allowed"]
    del observed["observation"]["escalation_history"]
    assert not authorize_action(observed, "escalate", ["inv-1"])["allowed"]
    assert scoped_rule_action(observed, ["inv-1"]) == "defer"


def test_prior_escalation_blocks_repeat_but_not_first_paid_refund():
    records = [{"id": "existing", "invoice_id": "inv-1"}]
    refund = state("Refund inv-1", records=records)
    assert authorize_action(refund, "refund", ["inv-1"])["allowed"]
    review = state("Escalate inv-1", status="disputed", records=records)
    assert not authorize_action(review, "escalate", ["inv-1"])["allowed"]
    assert scoped_rule_action(review, ["inv-1"], escalation_history=True) == "no_action"


def test_refund_request_does_not_authorize_different_operation_on_disputed_invoice():
    observed = state("Refund inv-1", status="disputed")
    assert not authorize_action(observed, "refund", ["inv-1"])["allowed"]
    assert not authorize_action(observed, "escalate", ["inv-1"])["allowed"]
    assert scoped_rule_action(observed, ["inv-1"]) == "defer"


def test_clear_scoped_resolution_selects_operation_from_actual_status():
    assert scoped_rule_action(state("Resolve the billing issue for inv-1"), ["inv-1"]) == "refund"
    assert (
        scoped_rule_action(
            state("Resolve the billing issue for inv-1", status="disputed"), ["inv-1"]
        )
        == "escalate"
    )
