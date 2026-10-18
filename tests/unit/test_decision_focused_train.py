"""Selection rewards safe coverage and rejects partition contamination."""

import copy

import pytest

from experiments.decision_model import data
from experiments.decision_model.focused_train import coverage_score, validate_partitions


def prediction(action="refund", gold="refund", confidence=0.9, error=None):
    return {"action": action, "label_action": gold, "confidence": confidence, "error": error}


def partitions():
    return [
        {
            "split": split,
            "group_id": f"family-{split}",
            "options": data.OPTIONS,
            "label": data.ACTIONS.index("refund"),
            "label_action": "refund",
        }
        for split in ("train", "dev", "calibration")
    ]


def test_coverage_penalizes_confident_incorrect_no_action():
    bad = [prediction("no_action", "refund")]
    safe_defer = [prediction("defer", "refund")]
    assert coverage_score(bad) == (-1, 0, 0)
    assert coverage_score(safe_defer) > coverage_score(bad)


def test_coverage_prefers_correct_direct_decisions_without_buying_errors():
    assert coverage_score([prediction()]) > coverage_score([prediction(confidence=0.69)])
    assert coverage_score([prediction(), prediction("refund", "escalate")]) < coverage_score(
        [prediction("defer", "refund")]
    )
    assert coverage_score([prediction("defer", "defer")]) == (0, 0, 1)
    assert coverage_score([prediction(confidence=0.7)]) == (0, 1, 1)


@pytest.mark.parametrize(
    "rows", [[], [prediction(error="timeout")], [prediction(), prediction(error="runtime")]]
)
def test_inference_errors_and_empty_results_invalidate_selection(rows):
    with pytest.raises(ValueError, match="complete and error-free"):
        coverage_score(rows)


def test_test_rows_and_missing_split_are_rejected():
    rows = partitions()
    for bad in ([*rows, {**rows[0], "split": "test"}], rows[:-1]):
        with pytest.raises(ValueError, match="TRAIN/DEV/CALIBRATION"):
            validate_partitions(bad)
    assert {k: len(v) for k, v in validate_partitions(rows).items()} == {
        "train": 1,
        "dev": 1,
        "calibration": 1,
    }


def test_cross_partition_family_overlap_is_rejected():
    rows = partitions()
    rows[1]["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError, match="Family appears"):
        validate_partitions(rows)


def test_pointer_label_follows_semantic_option_identity():
    rows = partitions()
    changed = copy.deepcopy(rows)
    changed[0]["options"] = list(reversed(changed[0]["options"]))
    with pytest.raises(ValueError, match="Option label mismatch"):
        validate_partitions(changed)
    changed[0]["label"] = next(
        i for i, option in enumerate(changed[0]["options"]) if option[0] == "refund"
    )
    assert len(validate_partitions(changed)["train"]) == 1
