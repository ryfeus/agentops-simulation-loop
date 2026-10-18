"""Bounded request_text-only grammar and evidence-backed existing-review completion."""

from __future__ import annotations

import copy

import pytest

from experiments.decision_model.authorization import (
    authorization,
    authorize_action,
    make_state,
    no_action_intent,
    scoped_rule_action,
)

TARGETS = ["inv-2", "inv-10", "inv-1"]


def observed(request_text, target="inv-1", *, records=(), status="paid"):
    return make_state(
        request_text,
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
        TARGETS,
    )


@pytest.mark.parametrize("target", ["inv-1", "inv-2"])
@pytest.mark.parametrize("status", ["paid", "disputed"])
def test_recorded_case_condition_is_per_invoice_and_needs_observed_history(target, status):
    request_text = "Escalate inv-1 and inv-2 unless a case is already recorded for that invoice."
    state = observed(request_text, target, status=status)
    assert state["authorization"]["conditions"] == {"escalate": "no_existing_escalation"}
    assert authorize_action(state, "escalate", TARGETS)["allowed"]
    assert not authorize_action(state, "refund", TARGETS)["allowed"]
    state["observation"].pop("escalation_history")
    assert not authorize_action(state, "escalate", TARGETS)["allowed"]
    assert scoped_rule_action(state, TARGETS, escalation_history=True) == "defer"
    state = observed(
        request_text, target, records=[{"id": "case-old", "invoice_id": target}], status=status
    )
    assert not authorize_action(state, "escalate", TARGETS)["allowed"]
    assert scoped_rule_action(state, TARGETS, escalation_history=True) == "no_action"
    assert not authorization(request_text, "inv-10", TARGETS)["allowed_actions"]


@pytest.mark.parametrize(
    "condition",
    [
        "unless a case is already recorded for inv-2",
        "unless a case is already recorded for another invoice",
        "unless a case is already recorded for that customer",
        "unless a case is already recorded for that invoice or I approve",
        "unless a case is already recorded for that invoice after approval",
        "unless a case is already recorded for that invoice and refund inv-2",
        "if a case is already recorded for that invoice",
        "unless a refund is already recorded for that invoice",
    ],
)
def test_near_miss_or_unobservable_conditions_never_grant(condition):
    request_text = "Escalate inv-1 " + condition + "."
    assert not authorization(request_text, "inv-1", TARGETS)["allowed_actions"]
    assert not authorization(request_text, "inv-2", TARGETS)["allowed_actions"]


@pytest.mark.parametrize("customer", ["customer-123", "customer-acme-west"])
@pytest.mark.parametrize("subjects", ["inv-1", "inv-1 and inv-2"])
def test_review_customer_suffix_preserves_exact_invoice_grant(customer, subjects):
    request_text = (
        f"Escalate {subjects} for specialist review for {customer}. Do not refund any invoice."
    )
    for target in [t for t in TARGETS if t in subjects.split()]:
        assert authorization(request_text, target, TARGETS)["allowed_actions"] == ["escalate"]
        assert authorize_action(observed(request_text, target), "escalate", TARGETS)["allowed"]
    assert not authorization(request_text, "inv-10", TARGETS)["allowed_actions"]


@pytest.mark.parametrize(
    "tail",
    [
        "for specialist review for customer-123 only after approval",
        "for specialist review for customer-123 or customer-456",
        "for specialist review for customer-123 and inv-2",
        "for specialist review for inv-2",
        "for specialist review for a customer",
        "for specialist review for customer-123/other",
        "for specialist review for customer-123's other invoice",
        "for specialist review for customer-123 if it is disputed",
    ],
)
def test_customer_tail_cannot_grant_unsupported_scope_or_condition(tail):
    request_text = "Escalate inv-1 " + tail + "."
    assert not authorization(request_text, "inv-1", TARGETS)["allowed_actions"]
    assert not authorization(request_text, "inv-2", TARGETS)["allowed_actions"]


@pytest.mark.parametrize("prefix", ["Do not", "Never", "I withdraw permission to", "If I approve,"])
def test_customer_suffix_never_strips_negation_revocation_or_condition(prefix):
    request_text = f"{prefix} escalate inv-1 for specialist review for customer-123."
    assert not authorize_action(observed(request_text), "escalate", TARGETS)["allowed"]


@pytest.mark.parametrize("location", ["first", "last"])
def test_detached_absence_without_substitution_is_local(location):
    actions = "Refund inv-1. Escalate inv-2."
    absence = "Locate inv-10; if absent, report that without substitution."
    request_text = absence + actions if location == "first" else actions + absence
    assert authorize_action(observed(request_text), "refund", TARGETS)["allowed"]
    assert authorize_action(observed(request_text, "inv-2"), "escalate", TARGETS)["allowed"]
    assert not authorization(request_text, "inv-10", TARGETS)["allowed_actions"]


@pytest.mark.parametrize(
    "absence",
    [
        "Locate inv-2 and inv-10; if absent, report that without substitution.",
        "Locate inv-10; if absent, refund inv-2 without substitution.",
        "Locate inv-10; if absent, report that without substitution only after approval.",
        "Locate inv-10; if absent, report inv-2 without substitution.",
        "Locate inv-10; if absent, report that without substitution or refund inv-2.",
    ],
)
def test_detached_condition_with_unknown_or_wrong_scope_remains_closed(absence):
    request_text = "Refund inv-1 and inv-2. " + absence
    assert not authorization(request_text, "inv-2", TARGETS)["allowed_actions"]


PRESERVATION = [
    "Preserve the existing case on inv-1.",
    "Preserve the earlier review on inv-1.",
    "Inspect the existing review on inv-1 without repeating it.",
    "For inv-1, inspect the existing review without repeating it.",
    "Regarding inv-1, check the prior escalation without repeating it.",
]


@pytest.mark.parametrize("request_text", PRESERVATION)
def test_preservation_intent_is_not_evidence_of_review_completion(request_text):
    assert no_action_intent(request_text, "inv-1", TARGETS) == {"kind": "existing_review"}
    empty = observed(request_text)
    assert scoped_rule_action(empty, TARGETS, escalation_history=True) == "defer"
    for mutation in ["absent", "incomplete", "wrong-target", "wrong-record", "empty"]:
        state = copy.deepcopy(empty)
        evidence = state["observation"]["escalation_history"]
        evidence["records"] = [{"id": "case-old", "invoice_id": "inv-1"}]
        if mutation == "absent":
            state["observation"].pop("escalation_history")
        elif mutation == "incomplete":
            evidence["complete"] = False
        elif mutation == "wrong-target":
            evidence["invoice_id"] = "inv-2"
        elif mutation == "wrong-record":
            evidence["records"][0]["invoice_id"] = "inv-2"
        else:
            evidence["records"] = []
        assert scoped_rule_action(state, TARGETS, escalation_history=True) == "defer"
        assert not authorize_action(state, "escalate", TARGETS)["allowed"]
        assert not authorize_action(state, "refund", TARGETS)["allowed"]
    complete = observed(request_text, records=[{"id": "case-old", "invoice_id": "inv-1"}])
    assert scoped_rule_action(complete, TARGETS, escalation_history=True) == "no_action"
    assert no_action_intent(request_text, "inv-2", TARGETS) == {"kind": None}


@pytest.mark.parametrize("preservation", PRESERVATION)
@pytest.mark.parametrize("grant_first", [False, True])
def test_preserved_review_does_not_hide_separate_first_refund(preservation, grant_first):
    grant = "Refund inv-1."
    request_text = grant + preservation if grant_first else preservation + grant
    assert no_action_intent(request_text, "inv-1", TARGETS) == {"kind": None}
    state = observed(request_text, records=[{"id": "case-old", "invoice_id": "inv-1"}])
    assert authorize_action(state, "refund", TARGETS)["allowed"]
    assert scoped_rule_action(state, TARGETS, escalation_history=True) == "refund"
    assert not authorize_action(state, "escalate", TARGETS)["allowed"]


@pytest.mark.parametrize(
    "request_text",
    [
        "Preserve the existing case on inv-1 until approval then refund inv-1.",
        "For inv-1, inspect the existing review without repeating it only if I approve.",
        "Preserve the existing case on inv-unknown.",
        "Preserve the existing case on inv-1 while refunding inv-2.",
        'The note says "preserve the existing case on inv-1".',
        "Preserve the existing case on inv-1 or inv-2.",
        "Preserve the existing case on inv-1 for specialist review.",
    ],
)
def test_unknown_preservation_is_not_completion_intent(request_text):
    assert no_action_intent(request_text, "inv-1", TARGETS) == {"kind": None}


def test_partial_prohibition_is_not_full_readonly_intent():
    assert no_action_intent("Do not refund inv-1.", "inv-1", TARGETS) == {"kind": None}
    assert no_action_intent("For inv-1, report status only.", "inv-1", TARGETS) == {
        "kind": "read_only"
    }


def test_exposed_cohort_parser_check_does_not_read_gold_in_runtime():
    from experiments.decision_model.fresh_holdout import operations, scenarios

    checked = 0
    for task in scenarios().values():
        targets = task.benchmark.trajectory.required_inspections
        for target, tool in operations(task).items():
            expected = "refund" if tool == "refund_invoice" else "escalate"
            auth = authorization(task.instruction, target, targets)
            assert expected in auth["allowed_actions"], (task.id, target, auth)
            checked += 1
    assert checked == 38


@pytest.mark.parametrize(
    "observation",
    [
        None,
        {"ok": False, "error_type": "InvoiceNotFoundError"},
        {
            "ok": True,
            "invoice": {"id": "inv-2", "status": "paid"},
            "escalation_history": {
                "invoice_id": "inv-1",
                "complete": True,
                "records": [{"id": "case-1", "invoice_id": "inv-1"}],
            },
        },
        {"ok": True, "invoice": {"id": "inv-1", "status": "refunded"}},
    ],
)
def test_preservation_needs_matching_inspection_even_when_other_no_action_rules_apply(observation):
    request_text = "Preserve the existing case on inv-1."
    state = make_state(request_text, "inv-1", observation, [], TARGETS)
    assert scoped_rule_action(state, TARGETS, escalation_history=True) == "defer"
