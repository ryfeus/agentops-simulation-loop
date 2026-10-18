"""Independent synthetic grammar challenges; no held-out family inputs or gold."""

import pytest

from experiments.decision_model.authorization import authorization, authorize_action
from experiments.decision_model.history import make_state

BILLS = ["bill-amber", "bill-birch", "bill-copper", "bill-dune", "bill-ember"]


def allowed(request, target, targets=BILLS):
    return authorization(request, target, targets)["allowed_actions"]


def observed(request, target, *, status="paid"):
    return make_state(
        request,
        target,
        {
            "ok": True,
            "invoice": {"id": target, "status": status},
            "escalation_history": {"invoice_id": target, "complete": True, "records": []},
        },
        [],
    )


@pytest.mark.parametrize("joiner", [", but ", " but ", ", and "])
@pytest.mark.parametrize("customer", ["", "For customer-amber, "])
@pytest.mark.parametrize("grant,deny", [("refund", "escalate"), ("escalate", "refund")])
def test_direct_mixed_grant_prohibition_preserves_only_positive_action(
    joiner, customer, grant, deny
):
    request = f"{customer}{grant} bill-amber{joiner}do not {deny} bill-amber."
    assert allowed(request, "bill-amber") == [grant]
    for target in BILLS[1:]:
        assert allowed(request, target) == []
    assert authorize_action(observed(request, "bill-amber"), deny, BILLS)["allowed"] is False


@pytest.mark.parametrize("prefix", ["", "Please "])
@pytest.mark.parametrize("joiner", [", ", ", and "])
@pytest.mark.parametrize("target_order", [BILLS, list(reversed(BILLS))])
def test_comma_commands_with_distinct_customers_keep_exact_operations(prefix, joiner, target_order):
    request = joiner.join(
        [
            prefix + "refund bill-amber for customer-north-branch",
            "escalate bill-birch for customer-south-branch",
            "refund bill-copper for customer-east-branch",
        ]
    )
    assert allowed(request, "bill-amber", target_order) == ["refund"]
    assert allowed(request, "bill-birch", target_order) == ["escalate"]
    assert allowed(request, "bill-copper", target_order) == ["refund"]
    assert allowed(request, "bill-dune", target_order) == []


@pytest.mark.parametrize("reference", ["those four invoices", "those 4 invoices"])
@pytest.mark.parametrize("target_order", [BILLS, list(reversed(BILLS))])
def test_counted_prohibition_resolves_exact_previous_list_without_harness_order(
    reference, target_order
):
    request = (
        "Refund bill-ember. Escalate bill-dune, bill-amber, bill-copper, and bill-birch. "
        f"Do not refund {reference}."
    )
    for target in BILLS[:4]:
        assert allowed(request, target, target_order) == ["escalate"]
    assert allowed(request, "bill-ember", target_order) == ["refund"]


@pytest.mark.parametrize(
    "restriction",
    [
        "Do not refund those four invoices.",
        "Do not refund those 4 invoices.",
        "Do not refund those first four invoices.",
        "Do not refund the fourth invoice.",
        "Do not refund those five invoices.",
    ],
)
def test_unresolved_or_mismatched_count_cannot_leave_refund_authority(restriction):
    request = "Refund bill-amber and bill-birch. " + restriction
    for target in BILLS[:2]:
        assert allowed(request, target) == []


def test_intervening_explicit_mention_invalidates_counted_list_antecedent():
    request = (
        "Refund bill-amber, bill-birch, bill-copper, and bill-dune. "
        "Inspect bill-ember. Do not refund those four invoices."
    )
    for target in BILLS[:4]:
        assert allowed(request, target) == []


@pytest.mark.parametrize("operation,other", [("refund", "escalate"), ("escalate", "refund")])
@pytest.mark.parametrize("verb", ["Withdraw", "Rescind", "Revoke"])
@pytest.mark.parametrize("withdrawal_first", [False, True])
def test_named_operation_withdrawal_cannot_cancel_independent_action(
    operation, other, verb, withdrawal_first
):
    grant = "Refund bill-amber and bill-birch. Escalate bill-amber and bill-birch. "
    withdrawal = f"{verb} permission to {operation} bill-amber. "
    request = withdrawal + grant if withdrawal_first else grant + withdrawal
    assert allowed(request, "bill-amber") == [other]
    assert allowed(request, "bill-birch") == ["escalate", "refund"]


@pytest.mark.parametrize(
    "restriction",
    [
        "Withdraw permission for bill-amber.",
        "Cancel that request for bill-amber.",
        "Withdraw the refund or escalation request for bill-amber.",
        "Withdraw the refund and escalation requests for bill-amber.",
        "Withdraw the refund-ish request for bill-amber.",
        "Withdraw the transfer request for bill-amber.",
        "Withdraw the permission about refund processing for bill-amber.",
        "Withdraw permission to refund bill-amber if approved.",
        "Withdraw permission to refund bill-amber or escalate bill-amber.",
        "Withdraw permission to refund bill-amber and resolve bill-amber.",
        "Withdraw permission to refund bill-amber for a customer.",
    ],
)
def test_unknown_or_multi_operation_withdrawal_stays_fail_closed(restriction):
    request = "Refund bill-amber. Escalate bill-amber. " + restriction
    assert allowed(request, "bill-amber") == []


@pytest.mark.parametrize(
    "request_text",
    [
        'The draft says "refund bill-amber, refund bill-birch, and refund bill-copper".',
        "The draft says 'refund bill-amber, refund bill-birch, and refund bill-copper'.",
        "`refund bill-amber, refund bill-birch, and refund bill-copper` is an example.",
        "The customer previously wrote refund bill-amber, refund bill-birch, "
        "and refund bill-copper.",
        "Maybe refund bill-amber, refund bill-birch, and refund bill-copper.",
        "Do not refund bill-amber, refund bill-birch, or refund bill-copper.",
        "Never refund bill-amber, refund bill-birch, and refund bill-copper.",
        "If I approve, refund bill-amber, refund bill-birch, and refund bill-copper.",
        "Refund bill-amber, refund bill-birch, and refund bill-copper only after I approve.",
        "Refund bill-amber if approved, refund bill-birch if approved, "
        "and refund bill-copper if approved.",
        "Do not refund bill-amber unless approved, but refund bill-birch only if approved.",
        "Refund bill-amber or bill-birch, refund bill-copper or bill-dune.",
    ],
)
def test_comma_split_never_promotes_quoted_narrated_uncertain_or_conditional_text(request_text):
    for target in BILLS:
        assert allowed(request_text, target) == [], (request_text, target)


@pytest.mark.parametrize(
    "request_text",
    [
        "If approved, escalate bill-amber, but do not refund bill-amber.",
        "Escalate bill-amber, but do not refund bill-amber until I approve.",
        'The instruction was "escalate bill-amber, but do not refund bill-amber".',
        "Perhaps escalate bill-amber, but do not refund bill-amber.",
    ],
)
def test_mixed_clause_repair_does_not_erase_governing_condition_or_narration(request_text):
    assert allowed(request_text, "bill-amber") == []


def test_independent_guard_still_requires_observed_paid_refund_state():
    request = "Refund bill-amber, escalate bill-birch for specialist review."
    assert allowed(request, "bill-amber") == ["refund"]
    state = observed(request, "bill-amber", status="disputed")
    assert not authorize_action(state, "refund", BILLS)["allowed"]


@pytest.mark.parametrize("joiner", ["and", "or"])
def test_counted_restriction_plus_explicit_extra_object_cannot_drop_counted_targets(joiner):
    request_text = (
        "Refund bill-ember. Refund bill-amber, bill-birch, bill-copper, and bill-dune. "
        f"Do not refund those four invoices {joiner} bill-ember."
    )
    for target in BILLS:
        assert allowed(request_text, target) == []


@pytest.mark.parametrize(
    "prefix",
    ["The customer wrote", "Earlier I requested", "For example:", "The proposed draft is"],
)
@pytest.mark.parametrize("joiner", [", ", ", and "])
def test_nonimperative_prefix_cannot_be_discarded_by_command_splitting(prefix, joiner):
    request_text = f"{prefix} refund bill-amber{joiner}refund bill-birch."
    for target in BILLS:
        assert allowed(request_text, target) == []
