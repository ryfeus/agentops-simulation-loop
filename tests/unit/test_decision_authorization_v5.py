"""Synthetic authorization grammar regressions; no held-out tasks or gold inputs."""

import copy

import pytest

from experiments.decision_model.authorization import (
    authorization,
    authorize_action,
    make_state,
    no_action_intent,
)

TARGETS = ["bill-cobalt", "bill-cedar", "bill-cobalt-long"]


def observed(request, target, *, records=()):
    return make_state(
        request,
        target,
        {
            "ok": True,
            "invoice": {"id": target, "status": "paid"},
            "escalation_history": {
                "invoice_id": target,
                "complete": True,
                "records": list(records),
            },
        },
        [],
        TARGETS,
    )


@pytest.mark.parametrize("conjunction", ["and", "or", ", bill-cobalt-long, or"])
@pytest.mark.parametrize("verb,other", [("refund", "escalate"), ("escalate", "refund")])
def test_negative_invoice_object_lists_preserve_other_operation(conjunction, verb, other):
    request = (
        "Refund bill-cobalt, bill-cedar, and bill-cobalt-long. "
        "Escalate bill-cobalt, bill-cedar, and bill-cobalt-long. "
        f"Do not {verb} bill-cobalt {conjunction} bill-cedar."
    )
    for target in TARGETS[:2]:
        auth = authorization(request, target, list(reversed(TARGETS)))
        assert auth["allowed_actions"] == [other]
        assert not authorize_action(observed(request, target), verb, TARGETS)["allowed"]
        assert authorize_action(observed(request, target), other, TARGETS)["allowed"]
        assert no_action_intent(request, target, TARGETS) == {"kind": None}


@pytest.mark.parametrize("conjunction", ["and", "or"])
@pytest.mark.parametrize("order", [False, True])
def test_negative_predicates_keep_different_target_scopes(conjunction, order):
    predicates = ["refund bill-cobalt or bill-cobalt-long", "escalate bill-cedar"]
    if order:
        predicates.reverse()
    request = (
        "Refund bill-cobalt, bill-cedar, and bill-cobalt-long. "
        "Escalate bill-cobalt, bill-cedar, and bill-cobalt-long. "
        "Never " + f" {conjunction} ".join(predicates) + "."
    )
    for target in TARGETS:
        assert authorization(request, target, TARGETS)["allowed_actions"] == [
            "refund" if target == "bill-cedar" else "escalate"
        ]


def test_shared_negative_verbs_bind_explicit_final_object_not_other_invoice():
    request = (
        "Refund bill-cobalt and bill-cedar. Escalate bill-cobalt and bill-cedar. "
        "Do not refund or escalate bill-cobalt."
    )
    assert not authorization(request, "bill-cobalt", TARGETS)["allowed_actions"]
    assert no_action_intent(request, "bill-cobalt", TARGETS) == {"kind": "read_only"}
    assert authorization(request, "bill-cedar", TARGETS)["allowed_actions"] == [
        "escalate",
        "refund",
    ]


def test_repeated_negation_preserves_mixed_operation_scopes():
    request = (
        "Refund bill-cobalt and bill-cedar. Escalate bill-cobalt and bill-cedar. "
        "Do not refund bill-cobalt and do not escalate bill-cedar."
    )
    assert authorization(request, "bill-cobalt", TARGETS)["allowed_actions"] == ["escalate"]
    assert authorization(request, "bill-cedar", TARGETS)["allowed_actions"] == ["refund"]


def test_unknown_condition_remains_attached_to_negative_history_scope():
    request = (
        "Refund bill-cedar. Do not repeat a recorded refund, and leave bill-cobalt "
        "unchanged if the necessary refund history is unavailable."
    )
    assert authorization(request, "bill-cedar", TARGETS)["allowed_actions"] == ["refund"]
    assert not authorization(request, "bill-cobalt", TARGETS)["allowed_actions"]


def test_positive_alternative_object_list_never_becomes_two_write_grants():
    request = "Refund bill-cobalt or bill-cedar for customer-451."
    for target in TARGETS:
        assert not authorization(request, target, TARGETS)["allowed_actions"]


@pytest.mark.parametrize(
    "negative",
    [
        "Do not refund bill-cobalt or transfer funds for bill-cobalt.",
        "Do not refund bill-cobalt or proceed.",
        "Never do that.",
        "I withdraw permission for bill-cobalt.",
    ],
)
def test_unknown_negative_and_withdrawal_stay_closed(negative):
    request = "Refund bill-cobalt. Escalate bill-cobalt. " + negative
    assert not authorization(request, "bill-cobalt", TARGETS)["allowed_actions"]


LOCAL_CONDITION = "only where that particular invoice does not yet have a review recorded"


@pytest.mark.parametrize("joiner", [" ", ", but "])
@pytest.mark.parametrize("target", TARGETS[:2])
def test_local_review_condition_requires_complete_matching_empty_history(joiner, target):
    request = "Escalate bill-cobalt and bill-cedar for specialist review" + joiner + LOCAL_CONDITION
    state = observed(request, target)
    assert state["authorization"]["conditions"] == {"escalate": "no_existing_escalation"}
    assert authorize_action(state, "escalate", TARGETS)["allowed"]
    assert not authorize_action(state, "refund", TARGETS)["allowed"]
    assert not authorization(request, "bill-cobalt-long", TARGETS)["allowed_actions"]
    for mutation in ["absent", "incomplete", "wrong-invoice", "wrong-record", "existing"]:
        changed = copy.deepcopy(state)
        history = changed["observation"]["escalation_history"]
        if mutation == "absent":
            changed["observation"].pop("escalation_history")
        elif mutation == "incomplete":
            history["complete"] = False
        elif mutation == "wrong-invoice":
            history["invoice_id"] = "bill-cobalt-long"
        else:
            history["records"] = [
                {
                    "id": "case-existing",
                    "invoice_id": "bill-cobalt-long" if mutation == "wrong-record" else target,
                }
            ]
        assert not authorize_action(changed, "escalate", TARGETS)["allowed"], mutation


@pytest.mark.parametrize(
    "condition",
    [
        "only where that customer does not yet have a review recorded",
        "only where another invoice does not yet have a review recorded",
        "only where bill-cedar does not yet have a review recorded",
        "only where that particular invoice has a review recorded",
        "only where that particular invoice does not yet have a refund recorded",
        LOCAL_CONDITION + " or I approve",
        LOCAL_CONDITION + " after approval",
        LOCAL_CONDITION + " and refund bill-cedar",
    ],
)
def test_unsupported_conditions_invalidate_even_an_earlier_grant(condition):
    request = f"Escalate bill-cobalt. Escalate bill-cobalt {condition}."
    assert not authorization(request, "bill-cobalt", TARGETS)["allowed_actions"]
    assert not authorization(request, "bill-cedar", TARGETS)["allowed_actions"]


def test_history_condition_never_authorizes_refund_or_overrides_withdrawal():
    refund = "Refund bill-cobalt " + LOCAL_CONDITION
    assert not authorization(refund, "bill-cobalt", TARGETS)["allowed_actions"]
    request = "Escalate bill-cobalt " + LOCAL_CONDITION + ". I withdraw permission for bill-cobalt."
    assert not authorization(request, "bill-cobalt", TARGETS)["allowed_actions"]


@pytest.mark.parametrize("customer", ["customer-451", "customer-north-branch"])
@pytest.mark.parametrize("subjects", ["bill-cobalt", "bill-cobalt and bill-cedar"])
def test_customer_qualified_refund_grants_only_direct_invoice_objects(customer, subjects):
    request = f"Refund {subjects} for {customer}. Escalate bill-cobalt-long."
    for target in TARGETS[:2]:
        assert authorize_action(observed(request, target), "refund", TARGETS)["allowed"] is (
            target in subjects.split()
        )
    assert not authorize_action(observed(request, "bill-cobalt-long"), "refund", TARGETS)["allowed"]


@pytest.mark.parametrize(
    "tail",
    [
        "for customer-451 after approval",
        "for customer-451 or customer-452",
        "for customer-451 and bill-cedar",
        "for customer-451's other invoice",
        "for customer-451/other",
        "for a customer",
        "for bill-cedar",
    ],
)
def test_customer_qualifier_cannot_supply_unknown_scope_or_authority(tail):
    request = "Refund bill-cobalt " + tail
    for target in TARGETS:
        assert not authorization(request, target, TARGETS)["allowed_actions"]


@pytest.mark.parametrize("withdrawal_first", [False, True])
def test_customer_qualified_refund_preserves_independent_withdrawal(withdrawal_first):
    grant = "Refund bill-cobalt for customer-451. Escalate bill-cedar for customer-452. "
    withdrawal = "I withdraw permission for bill-cedar. "
    request = withdrawal + grant if withdrawal_first else grant + withdrawal
    assert authorization(request, "bill-cobalt", TARGETS)["allowed_actions"] == ["refund"]
    assert not authorization(request, "bill-cedar", TARGETS)["allowed_actions"]
