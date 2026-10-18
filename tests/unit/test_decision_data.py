"""Decision data must preserve observation timing, family isolation, and option labels."""

import copy
import json

import pytest

from experiments.decision_model.data import (
    ACTIONS,
    extract_records,
    make_state,
    remap_options,
    rule_action,
    validate_records,
)


def demo(task="train-task", *, status="paid", action="refund_invoice"):
    call = {"name": "get_invoice", "arguments": {"invoice_id": "inv-1"}}
    observation = {"ok": True, "invoice": {"id": "inv-1", "status": status}}
    messages = [{"role": "user", "content": "Please refund inv-1."}]
    calls = [call]
    results = [observation]
    if action:
        calls.append({"name": action, "arguments": {"invoice_id": "inv-1", "reason": "Resolve"}})
        key = "refund" if action == "refund_invoice" else "escalation"
        results.append({"ok": True, key: {"invoice_id": "inv-1", "success": True}})
    for call, result in zip(calls, results, strict=True):
        messages.extend(
            [
                {"role": "assistant", "tool_calls": [{"function": call}]},
                {"role": "tool", "name": call["name"], "content": json.dumps(result)},
            ]
        )
    messages.append({"role": "assistant", "content": "Done."})
    return {
        "task_id": task,
        "complete": True,
        "truncated": False,
        "source": "executed-billing-tools",
        "reward": 1.0,
        "verifier_components": {"reward": 1.0},
        "tool_calls": calls,
        "messages": messages,
    }


def meta(*tasks, group="paid-refund-direct"):
    return {task: {"prototype_id": group, "family_id": task + "-family"} for task in tasks}


def test_observed_timing_no_future_or_metadata():
    records, _ = extract_records([demo()], meta("train-task"))
    inspect, missing, completed = records
    before, after = json.loads(inspect["state"]), json.loads(completed["state"])
    assert before["prior_actions"] == []
    assert json.loads(missing["state"])["observation"] is None
    assert after["observation"] == before["observation"]  # Never invent refunded status.
    assert after["prior_actions"][0]["result"]["refund"]["success"] is True
    assert inspect["label_action"] == "refund"
    assert completed["label_action"] == "no_action"
    for key in ("task_id", "prototype_id", "reward", "verifier_components", "split"):
        assert key not in before


def test_hidden_prior_escalation_is_defer_not_invented_noop():
    rows, _ = extract_records([demo(status="disputed", action=None)], meta("train-task"))
    assert rows[0]["label_action"] == "defer"
    assert rows[0]["label_basis"] == "teacher_noop_requires_unobserved_history"
    assert json.loads(rows[0]["state"])["prior_actions"] == []


def test_siblings_and_augmentations_share_split():
    rows, groups = extract_records([demo("one"), demo("two")], meta("one", "two"))
    assert groups == {"paid-refund-direct": "dev"}
    assert {row["split"] for row in rows} == {"dev"}
    rows[0]["split"] = "train"
    with pytest.raises(ValueError, match="split leakage"):
        validate_records(rows)


def test_option_permutation_remaps_action():
    rows, _ = extract_records([demo()], meta("train-task"))
    for order in ([3, 2, 1, 0], [1, 2, 3, 0]):
        changed = remap_options(rows[0], list(order))
        assert changed["options"][changed["label"]][0] == "refund"
    with pytest.raises(ValueError, match="permutation"):
        remap_options(rows[0], [0, 0, 2, 3])


@pytest.mark.parametrize("change", [{"reward": 0.0}, {"complete": False}, {"truncated": True}])
def test_unverified_demos_rejected(change):
    record = demo()
    record.update(change)
    with pytest.raises(ValueError, match="unverified"):
        extract_records([record], meta("train-task"))


def test_historical_nontrain_task_rejected():
    with pytest.raises(ValueError, match="non-training"):
        extract_records([demo("historical-test")], meta("train-task"))


def test_tool_evidence_mismatch_rejected():
    record = demo()
    record["tool_calls"] = copy.deepcopy(record["tool_calls"])
    record["tool_calls"][0]["arguments"]["invoice_id"] = "wrong"
    with pytest.raises(ValueError, match="trajectory"):
        extract_records([record], meta("train-task"))


def test_rule_uses_observed_success_not_attempted_action():
    state = make_state(
        "Refund inv-1",
        "inv-1",
        {
            "ok": True,
            "invoice": {"id": "inv-1", "status": "paid"},
        },
        [{"name": "refund_invoice", "arguments": {"invoice_id": "inv-1"}, "result": {"ok": False}}],
    )
    assert rule_action(state) == "refund"
    state["prior_actions"][0]["result"] = {
        "ok": True,
        "refund": {"invoice_id": "inv-1", "success": True},
    }
    assert rule_action(state) == "no_action"
    state["prior_actions"][0]["result"]["refund"]["invoice_id"] = "inv-2"
    assert rule_action(state) == "refund"


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (None, "defer"),
        ({"ok": False, "error_type": "InvoiceNotFoundError"}, "no_action"),
        ({"ok": False, "error_type": "TimeoutError"}, "defer"),
        ({"ok": True, "invoice": {"id": "inv-1", "status": "disputed"}}, "escalate"),
        ({"ok": True, "invoice": {"id": "inv-1", "status": "refunded"}}, "no_action"),
        ({"ok": True, "invoice": {"id": "inv-2", "status": "paid"}}, "defer"),
    ],
)
def test_rule_boundaries(observation, expected):
    assert rule_action(make_state("Refund inv-1", "inv-1", observation, [])) == expected


def test_context_invoice_is_not_refunded():
    request = "Please inspect both invoices and refund inv-1 while protecting inv-2."
    state = make_state(
        request,
        "inv-2",
        {
            "ok": True,
            "invoice": {"id": "inv-2", "status": "paid"},
        },
        [],
    )
    assert rule_action(state) == "no_action"


def test_labels_independent_of_rule_parser():
    record = demo()
    record["messages"][0]["content"] = "Please process the specified bill, inv-1, only."
    rows, _ = extract_records([record], meta("train-task"))
    assert rows[0]["label_action"] == "refund"
    assert rule_action(json.loads(rows[0]["state"])) == "defer"
    assert set(a[0] for a in rows[0]["options"]) == set(ACTIONS)


def test_conflicting_observable_states_defer_symmetrically():
    rows, _ = extract_records(
        [
            demo("one", status="disputed", action="escalate_dispute"),
            demo("two", status="disputed", action=None),
        ],
        meta("one", "two"),
    )
    inspections = [row for row in rows if row["boundary"] == "after_inspection"]
    assert len(inspections) == 2
    assert {row["label_action"] for row in inspections} == {"defer"}
    assert {row["label_basis"] for row in inspections} == {"conflicting_observable_labels"}
    assert {row["original_label_action"] for row in inspections} == {"escalate", "defer"}
