"""Legacy risk correction is independent of predictions and cannot change gold."""

import copy

import pytest

from experiments.decision_model import authorization, data, focused_calibrate, model


def fixture():
    state = authorization.make_state(
        "Escalate inv-1 unless already escalated. Do not refund inv-1.",
        "inv-1",
        {
            "ok": True,
            "invoice": {"id": "inv-1", "status": "paid"},
            "escalation_history": {"invoice_id": "inv-1", "complete": True, "records": []},
        },
        [],
        ["inv-1"],
    )
    gold = {
        "example_id": "task:initial",
        "task_id": "task",
        "group_id": "family",
        "split": "calibration",
        "label_action": "escalate",
        "prohibited_actions": ["escalate", "refund"],
        "input_state": state,
        "state": data.serialize_state(state),
    }
    predicted = {
        k: copy.deepcopy(gold[k])
        for k in ("example_id", "group_id", "split", "label_action", "prohibited_actions")
    }
    predicted.update(
        action="escalate",
        probabilities={"escalate": 0.97, "refund": 0.01, "defer": 0.01, "no_action": 0.01},
        confidence=0.97,
        error=None,
    )
    task = {
        "split": "calibration",
        "scenario_sha256": "hash",
        "required_inspections": ["inv-1"],
        "expected_operations": {"inv-1": "escalate_dispute"},
    }
    demo = {
        "split": "calibration",
        "scenario_sha256": "hash",
        "complete": True,
        "reward": 1,
        "verifier_components": {"reward": 1},
        "tool_observations": [
            {
                "name": "escalate_dispute",
                "arguments": {"invoice_id": "inv-1"},
                "result": {"ok": True},
            }
        ],
    }
    return [predicted], [gold], {"task": task}, {"task": demo}


def test_only_contradictory_gold_flag_removed_originals_unchanged():
    args = fixture()
    original = copy.deepcopy(args)
    corrected, audit = focused_calibrate.reconcile(*args)
    assert args == original
    assert corrected[0]["prohibited_actions"] == ["refund"]
    assert len(audit) == 1
    assert all(corrected[0][k] == args[0][0][k] for k in args[0][0] if k != "prohibited_actions")
    assert model.fit_calibration(args[0], {})["accepted_count"] == 0
    assert model.fit_calibration(corrected, {})["accepted_count"] == 1


def test_wrong_prediction_does_not_determine_correction_or_get_accepted():
    args = fixture()
    args[0][0].update(
        action="refund",
        probabilities={"refund": 0.97, "escalate": 0.01, "defer": 0.01, "no_action": 0.01},
    )
    corrected, audit = focused_calibrate.reconcile(*args)
    assert corrected[0]["prohibited_actions"] == ["refund"] and len(audit) == 1
    assert model.fit_calibration(corrected, {})["accepted_count"] == 0


@pytest.mark.parametrize("bad", ["test", "coverage", "gold", "evidence", "prediction"])
def test_correction_rejects_unverified_or_changed_inputs(bad):
    predicted, gold, tasks, demonstrations = fixture()
    if bad == "test":
        gold[0]["split"] = "test"
    elif bad == "coverage":
        predicted = []
    elif bad == "gold":
        gold[0]["label_action"] = "refund"
    elif bad == "evidence":
        demonstrations["task"]["tool_observations"] = []
    else:
        predicted[0]["error"] = "inference failed"
    with pytest.raises(ValueError):
        focused_calibrate.reconcile(predicted, gold, tasks, demonstrations)
