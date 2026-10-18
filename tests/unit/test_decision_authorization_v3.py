"""Request-only scope repairs, including unseen ID/order and negative-clause variants."""

import pytest

from experiments.decision_model.authorization import (
    authorization,
    authorize_action,
    make_state,
    scoped_rule_action,
)


def observed(request_text, target, targets, *, status="paid", history=()):
    observation = {
        "ok": True,
        "invoice": {"id": target, "status": status},
    }
    if history is not None:
        observation["escalation_history"] = {
            "invoice_id": target,
            "complete": True,
            "records": list(history),
        }
    return make_state(request_text, target, observation, [], targets)


@pytest.mark.parametrize("article", ["", "a ", "the "])
def test_generic_duplicate_review_prohibition_preserves_first_review(article):
    request_text = (
        "Refund inv-1. Escalate inv-2. "
        f"Never refund a disputed invoice or create {article}duplicate review."
    )
    targets = ["inv-2", "inv-1"]
    refund = observed(request_text, "inv-1", targets)
    assert authorize_action(refund, "refund", targets)["allowed"]
    review = observed(request_text, "inv-2", targets, status="disputed")
    assert authorize_action(review, "escalate", targets)["allowed"]
    assert review["authorization"]["conditions"] == {"escalate": "no_existing_escalation"}
    review["observation"].pop("escalation_history")
    assert scoped_rule_action(review, targets, escalation_history=True) == "defer"
    review = observed(
        request_text,
        "inv-2",
        targets,
        status="disputed",
        history=[{"id": "case-1", "invoice_id": "inv-2"}],
    )
    assert not authorize_action(review, "escalate", targets)["allowed"]
    assert scoped_rule_action(review, targets, escalation_history=True) == "no_action"


def test_unknown_duplicate_negative_predicate_still_denies():
    request_text = "Escalate inv-1. Never create a duplicate review or transfer funds."
    assert not authorization(request_text, "inv-1", ["inv-1"])["allowed_actions"]


@pytest.mark.parametrize("first,second", [("inv-1", "inv-22"), ("bill-904", "bill-9")])
@pytest.mark.parametrize("exception_first", [False, True])
def test_readonly_nonrestriction_clause_is_scoped_without_granting(first, second, exception_first):
    exception = (
        f"The status-only inquiry for {first} does not restrict the separate action on {second}."
    )
    commands = f"Refund {first}. Refund {second}."
    request_text = exception + commands if exception_first else commands + exception
    targets = [second, first]
    assert authorization(request_text, first, targets)["status"] == "denied"
    assert authorization(request_text, second, targets)["allowed_actions"] == ["refund"]
    # Saying a separate action is unaffected never supplies missing write authority.
    assert not authorization(exception, second, targets)["allowed_actions"]
    assert not authorization(
        f"Refund {second}. Do not refund {second}. " + exception, second, targets
    )["allowed_actions"]


@pytest.mark.parametrize(
    "suffix",
    [
        "The status-only inquiry for inv-1 does not restrict the separate action on inv-2 "
        "only after approval.",
        "The status-only inquiry for inv-1 does not restrict the separate action on inv-unknown.",
        "The status-only inquiry for inv-1 does not forbid changing inv-2.",
    ],
)
def test_unknown_readonly_exceptions_do_not_create_authority(suffix):
    assert not authorization(suffix, "inv-2", ["inv-1", "inv-2"])["allowed_actions"]


@pytest.mark.parametrize("first,second", [("inv-1", "inv-22"), ("bill-904", "bill-9")])
@pytest.mark.parametrize("grant_first", [True, False])
def test_late_withdrawal_preserves_other_existing_grant_only(first, second, grant_first):
    grants = f"Refund {first} and {second}."
    withdrawal = (
        f"Correction: do not refund {first}; I withdraw that invoice's authorization, "
        f"while the refund of {second} remains authorized."
    )
    request_text = grants + withdrawal if grant_first else withdrawal + grants
    targets = [second, first]
    assert authorization(request_text, first, targets)["status"] == "denied"
    assert authorize_action(observed(request_text, second, targets), "refund", targets)["allowed"]
    assert not authorization(withdrawal, second, targets)["allowed_actions"]
    assert not authorization(f"Do not refund {second}. " + request_text, second, targets)[
        "allowed_actions"
    ]


@pytest.mark.parametrize("conjunction", ["while", "whereas", "but"])
def test_multiple_withdrawn_actions_do_not_capture_preserved_target(conjunction):
    request_text = (
        "Refund inv-1 and inv-3. Escalate inv-2. "
        "I withdraw permission to refund inv-1 and escalate inv-2, "
        f"{conjunction} the refund of inv-3 remains authorized."
    )
    targets = ["inv-3", "inv-2", "inv-1"]
    for target in ["inv-1", "inv-2"]:
        assert authorization(request_text, target, targets)["status"] == "denied"
    assert authorization(request_text, "inv-3", targets)["allowed_actions"] == ["refund"]


@pytest.mark.parametrize(
    "request_text",
    [
        "Refund inv-1 and inv-2. I withdraw permission, "
        "while the refund of inv-2 remains authorized.",
        "Refund inv-1 and inv-2. I withdraw it, while the refund of inv-2 remains authorized.",
        "Refund inv-1 and inv-2. Never do that, while the refund of inv-2 remains authorized.",
        "Refund inv-1 and inv-2. I withdraw inv-1 only after approval, "
        "while the refund of inv-2 remains authorized.",
        "Refund inv-1 and inv-2. Do not transfer inv-1, "
        "while the refund of inv-2 remains authorized only after approval.",
    ],
)
def test_unresolved_withdrawal_or_preservation_cannot_loosen_denials(request_text):
    assert not authorization(request_text, "inv-2", ["inv-1", "inv-2"])["allowed_actions"]


def test_duplicate_refund_constraint_does_not_invent_refund_history():
    request_text = "Refund inv-1. Never create a duplicate refund."
    state = observed(request_text, "inv-1", ["inv-1"])
    assert state["authorization"]["conditions"] == {"refund": "no_existing_refund"}
    assert scoped_rule_action(state, ["inv-1"]) == "defer"


def test_corrective_unknown_or_conditional_command_still_fails_closed():
    for request_text in [
        "Correction: transfer money for inv-1.",
        "Correction: refund inv-1 only after approval.",
        'The note says "Correction: refund inv-1".',
    ]:
        assert not authorization(request_text, "inv-1", ["inv-1"])["allowed_actions"]


def test_corrective_shared_negation_blocks_both_actual_write_guards():
    request_text = "Refund inv-1 and inv-2. Correction: do not refund inv-1 and inv-2."
    targets = ["inv-2", "inv-1"]
    for target in targets:
        assert not authorize_action(observed(request_text, target, targets), "refund", targets)[
            "allowed"
        ]


def test_disputed_refund_prohibition_preserves_requested_paid_refunds_and_reviews():
    request_text = (
        "Refund inv-1 and inv-2. Escalate inv-3 and inv-4. "
        "Do not turn either disputed invoice into a refund."
    )
    targets = ["inv-4", "inv-3", "inv-2", "inv-1"]
    for target in ["inv-1", "inv-2"]:
        state = observed(request_text, target, targets)
        assert authorize_action(state, "refund", targets)["allowed"]
        state["observation"]["invoice"]["status"] = "disputed"
        assert not authorize_action(state, "refund", targets)["allowed"]
    for target in ["inv-3", "inv-4"]:
        state = observed(request_text, target, targets, status="disputed")
        assert authorize_action(state, "escalate", targets)["allowed"]
        assert not authorize_action(state, "refund", targets)["allowed"]


@pytest.mark.parametrize("review_target", ["inv-1", "inv-2"])
def test_refund_preserves_prior_review_without_transferring_target_authority(review_target):
    request_text = f"Refund inv-1 and inv-2, retaining the earlier review on {review_target}."
    targets = ["inv-3", "inv-2", "inv-1"]
    for target in ["inv-1", "inv-2"]:
        state = observed(
            request_text, target, targets, history=[{"id": "case-old", "invoice_id": target}]
        )
        assert authorize_action(state, "refund", targets)["allowed"]
        assert not authorize_action(state, "escalate", targets)["allowed"]
    assert not authorization(request_text, "inv-3", targets)["allowed_actions"]


@pytest.mark.parametrize(
    "request_text",
    [
        "Refund inv-1, retaining the earlier review on inv-2.",
        "Refund inv-1, retaining the earlier review on inv-1 only after approval.",
        "Refund inv-1, retaining the earlier review on inv-1 while modifying inv-2.",
        "Refund inv-1, retaining the earlier refund on inv-1.",
        "Refund inv-1. Do not turn either invoice into a refund.",
        "Refund inv-1. Do not turn either disputed invoice into a refund or transfer funds.",
    ],
)
def test_unknown_preservation_or_negative_tails_remain_fail_closed(request_text):
    assert not authorization(request_text, "inv-1", ["inv-1", "inv-2"])["allowed_actions"]


@pytest.mark.parametrize("absence_first", [False, True])
def test_missing_invoice_report_condition_does_not_revoke_independent_review(absence_first):
    commands = "Refund inv-1. Escalate inv-2 and inv-3 unless already escalated."
    absence = "Locate inv-4; if absent, report that without substituting another bill."
    request_text = absence + commands if absence_first else commands + absence
    targets = ["inv-4", "inv-3", "inv-2", "inv-1"]
    assert authorization(request_text, "inv-1", targets)["allowed_actions"] == ["refund"]
    for target in ["inv-2", "inv-3"]:
        assert authorize_action(
            observed(request_text, target, targets, status="disputed"), "escalate", targets
        )["allowed"]
    assert not authorization(request_text, "inv-4", targets)["allowed_actions"]


@pytest.mark.parametrize(
    "suffix",
    [
        "Locate inv-3; only after I approve.",
        "Locate inv-3; if absent, refund inv-2.",
        "Locate inv-2 and inv-3; if absent, report that without substituting another bill.",
        "Locate inv-3; if absent, report that without substituting another bill unless I approve.",
    ],
)
def test_unknown_or_unresolved_absence_conditions_remain_fail_closed(suffix):
    request_text = "Refund inv-1 and inv-2. " + suffix
    assert not authorization(request_text, "inv-2", ["inv-1", "inv-2", "inv-3"])["allowed_actions"]


@pytest.mark.parametrize(
    "restriction",
    [
        "Inspect the existing case on inv-1 without repeating it or issuing a refund.",
        "Inspect the existing review on inv-1 without repeating it.",
        "Preserve the existing case on inv-1.",
    ],
)
def test_explicit_prior_case_observation_is_readonly_and_target_specific(restriction):
    request_text = restriction + "Refund inv-2. Escalate inv-3."
    targets = ["inv-3", "inv-2", "inv-1"]
    for status in ["paid", "disputed"]:
        state = observed(
            request_text,
            "inv-1",
            targets,
            status=status,
            history=[{"id": "case-1", "invoice_id": "inv-1"}],
        )
        assert scoped_rule_action(state, targets, escalation_history=True) == "no_action"
        for action in ["refund", "escalate"]:
            assert not authorize_action(state, action, targets)["allowed"]
    assert authorization(request_text, "inv-2", targets)["allowed_actions"] == ["refund"]
    assert authorization(request_text, "inv-3", targets)["allowed_actions"] == ["escalate"]


def test_unknown_existing_case_wording_does_not_become_automatic_no_action():
    request_text = "Preserve the existing case on inv-1 until approval then refund inv-1."
    state = observed(request_text, "inv-1", ["inv-1"])
    assert scoped_rule_action(state, ["inv-1"]) == "defer"


@pytest.mark.parametrize(
    "restriction",
    [
        "Preserve the existing case on inv-1.",
        "Inspect the existing case on inv-1 without repeating it.",
    ],
)
@pytest.mark.parametrize("restriction_first", [False, True])
def test_prior_case_restriction_does_not_cancel_separate_first_refund(
    restriction, restriction_first
):
    command = "Refund inv-1."
    request_text = restriction + command if restriction_first else command + restriction
    targets = ["inv-1"]
    state = observed(
        request_text, "inv-1", targets, history=[{"id": "case-old", "invoice_id": "inv-1"}]
    )
    assert authorize_action(state, "refund", targets)["allowed"]
    assert not authorize_action(state, "escalate", targets)["allowed"]
    assert scoped_rule_action(state, targets, escalation_history=True) == "refund"


def test_explicit_prior_case_refund_prohibition_still_overrides_grant():
    request_text = (
        "Refund inv-1. Inspect the existing case on inv-1 without repeating it or issuing a refund."
    )
    state = observed(
        request_text,
        "inv-1",
        ["inv-1"],
        history=[{"id": "case-1", "invoice_id": "inv-1"}],
    )
    assert not authorize_action(state, "refund", ["inv-1"])["allowed"]
    assert scoped_rule_action(state, ["inv-1"]) == "no_action"
