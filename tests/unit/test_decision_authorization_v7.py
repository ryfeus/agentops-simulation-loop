"""Invoice grammar repairs exercised against actual observed-state guard behavior."""

import pytest

from experiments.decision_model.authorization import authorization, authorize_action, make_state

TARGETS = [f"bill-{word}" for word in ("oak", "pine", "elm", "ash", "fir", "birch")]


def observed(request_text, target, *, status="paid", reviewed=False):
    return make_state(
        request_text,
        target,
        {
            "ok": True,
            "invoice": {"id": target, "status": status},
            "escalation_history": {
                "invoice_id": target,
                "complete": True,
                "records": [{"invoice_id": target, "id": "review"}] if reviewed else [],
            },
        },
        [],
        TARGETS,
    )


@pytest.mark.parametrize("joiner", [", but ", " but ", ", and "])
@pytest.mark.parametrize("prohibition", ["do not refund", "never refund", "must not refund"])
def test_mixed_grant_prohibition_keeps_disputed_review(joiner, prohibition):
    request_text = (
        f"For customer-north, escalate bill-oak for specialist review{joiner}"
        f"{prohibition} bill-oak."
    )
    state = observed(request_text, "bill-oak", status="disputed")
    assert state["authorization"]["allowed_actions"] == ["escalate"]
    assert authorize_action(state, "escalate", TARGETS)["allowed"]
    assert not authorize_action(state, "refund", TARGETS)["allowed"]
    assert not authorize_action(
        observed(request_text, "bill-oak", reviewed=True), "escalate", TARGETS
    )["allowed"]


@pytest.mark.parametrize("count", ["four", "4"])
def test_counted_queue_denial_does_not_spill_into_separate_refund(count):
    request_text = (
        "Escalate bill-oak, bill-pine, bill-elm, and bill-ash for specialist review "
        "only if each invoice has no escalation already recorded. "
        f"Do not refund those {count} invoices. Independently, refund bill-fir. "
        "Refund bill-birch. I withdraw permission to refund bill-birch."
    )
    for target in TARGETS[:4]:
        state = observed(request_text, target)
        assert state["authorization"]["conditions"] == {"escalate": "no_existing_escalation"}
        assert authorize_action(state, "escalate", TARGETS)["allowed"]
        assert not authorize_action(state, "refund", TARGETS)["allowed"]
    assert authorize_action(observed(request_text, "bill-fir", reviewed=True), "refund", TARGETS)[
        "allowed"
    ]
    assert not authorize_action(observed(request_text, "bill-birch"), "refund", TARGETS)["allowed"]


@pytest.mark.parametrize("withdrawn,retained", [("refund", "escalate"), ("escalate", "refund")])
@pytest.mark.parametrize("first", [False, True])
def test_action_withdrawal_preserves_only_explicit_other_operation(withdrawn, retained, first):
    grants = "Refund bill-oak and escalate bill-oak for specialist review. "
    withdrawal = f"I withdraw permission to {withdrawn} bill-oak. "
    request_text = withdrawal + grants if first else grants + withdrawal
    state = observed(request_text, "bill-oak")
    assert state["authorization"]["allowed_actions"] == [retained]
    assert authorize_action(state, retained, TARGETS)["allowed"]
    assert not authorize_action(state, withdrawn, TARGETS)["allowed"]
    assert not authorization(withdrawal, "bill-oak", TARGETS)["allowed_actions"]


def test_three_customer_commands_preserve_paid_refund_and_history_policy():
    request_text = (
        "Refund bill-oak for customer-north, refund bill-pine for customer-south, "
        "and refund bill-elm for customer-west."
    )
    for target in TARGETS[:3]:
        assert authorize_action(observed(request_text, target, reviewed=True), "refund", TARGETS)[
            "allowed"
        ]
        assert not authorize_action(
            observed(request_text, target, status="disputed"), "refund", TARGETS
        )["allowed"]
        assert not authorize_action(observed(request_text, target), "escalate", TARGETS)["allowed"]
    for target in TARGETS[3:]:
        assert not authorization(request_text, target, TARGETS)["allowed_actions"]


@pytest.mark.parametrize(
    "tail",
    [
        "if I approve",
        "only after approval",
        "or transfer money",
        "for an unknown customer",
        "for customer-north's other invoice",
        "and escalate bill-pine",
        "from the quoted note",
    ],
)
def test_unrecognized_withdrawal_never_narrows_revocation(tail):
    request_text = (
        f"Refund bill-oak. Escalate bill-oak. I withdraw permission to refund bill-oak {tail}."
    )
    assert not authorization(request_text, "bill-oak", TARGETS)["allowed_actions"]


@pytest.mark.parametrize(
    "quotation", [('"', '"'), ("'", "'"), ("`", "`"), ("“", "”"), ("\u2018", "\u2019")]
)
@pytest.mark.parametrize("separator", ["; ", ". ", "\n"])
def test_quoted_command_separators_never_promote_interior_command(quotation, separator):
    left, right = quotation
    request_text = (
        f"The note says {left}refund bill-oak{separator}refund bill-pine"
        f"{separator}refund bill-elm{right}. Refund bill-fir."
    )
    for target in TARGETS[:3]:
        assert not authorization(request_text, target, TARGETS)["allowed_actions"]
    assert authorization(request_text, "bill-fir", TARGETS)["allowed_actions"] == ["refund"]


def test_unterminated_quotation_and_quoted_condition_do_not_create_grant():
    for request_text in [
        'The note says "refund bill-oak; refund bill-pine. Refund bill-elm.',
        'Refund bill-oak "only after approval".',
        "The note says 'don't refund bill-oak; refund bill-pine; refund bill-elm'.",
    ]:
        for target in TARGETS[:3]:
            assert not authorization(request_text, target, TARGETS)["allowed_actions"]
