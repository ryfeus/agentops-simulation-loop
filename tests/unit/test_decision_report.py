"""Replay comparisons must not inflate coverage, correctness, or independent sample counts."""

import json

import pytest

from experiments.decision_model.data import ACTIONS
from experiments.decision_model.report import (
    generate_report,
    join_predictions,
    metric_summary,
    paired_bootstrap,
    slice_rows,
)


def row(key="one", action="refund", group="family-a", boundary="after_inspection"):
    return {
        "example_id": key,
        "task_id": "task-" + key,
        "group_id": group,
        "split": "test",
        "label_action": action,
        "prohibited_actions": [],
        "boundary": boundary,
    }


def prediction(key="one", action="refund", *, error=None, probabilities=None):
    return {
        "example_id": key,
        "action": action,
        "error": error,
        "probabilities": probabilities,
        "latency_ms": 10.0,
        "input_tokens": None,
    }


def test_coverage_matches_example_id_and_dataset_split():
    data = [row(), {**row("train"), "split": "train"}]
    joined, coverage = join_predictions(data, [prediction(), prediction("train")])
    assert len(joined) == 1
    assert coverage == {
        "expected": 1,
        "available": 1,
        "missing": 0,
        "filtered_non_test_predictions": 1,
    }
    # Prediction gold/split annotations never define the metric cohort or truth.
    joined, _ = join_predictions(
        data, [{**prediction(), "label_action": "defer", "split": "train"}]
    )
    assert metric_summary(joined)["accuracy"] == 1.0


def test_missing_unknown_duplicate_rejected():
    with pytest.raises(ValueError, match="Missing"):
        join_predictions([row(), row("two")], [prediction()])
    with pytest.raises(ValueError, match="Unknown"):
        join_predictions([row()], [prediction("unknown")])
    with pytest.raises(ValueError, match="Duplicate prediction"):
        join_predictions([row()], [prediction(), prediction()])
    with pytest.raises(ValueError, match="Duplicate dataset"):
        join_predictions([row(), row()], [prediction()])


def test_partial_requires_explicit_opt_in():
    joined, coverage = join_predictions([row(), row("two")], [prediction()], allow_partial=True)
    assert len(joined) == 1
    assert coverage["missing"] == 1


def test_error_defer_not_counted_correct_or_successful_abstention():
    summary = metric_summary([(row(action="defer"), prediction(action="defer", error="timeout"))])
    assert summary["accuracy"] == 0
    assert summary["errors"] == 1
    assert summary["defer_count"] == 0
    assert summary["fallback_needed_rate"] == 1
    assert summary["confusion_matrix"]["defer"]["error"] == 1
    assert summary["non_defer_accuracy"] is None


def test_abstention_accuracy_distinct_from_accepted_accuracy():
    summary = metric_summary(
        [
            (row("a", "refund"), prediction("a", "refund")),
            (row("b", "defer"), prediction("b", "defer")),
            (row("c", "refund"), prediction("c", "defer")),
        ]
    )
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["non_defer_accuracy"] == 1
    assert summary["non_defer_coverage"] == pytest.approx(1 / 3)
    assert summary["defer_rate"] == pytest.approx(2 / 3)
    assert summary["gold_defer_count"] == 1


def test_calibration_only_valid_probabilities():
    correct = {a: float(a == "refund") for a in ACTIONS}
    wrong = {a: float(a == "escalate") for a in ACTIONS}
    joined = [
        (row("a"), prediction("a", probabilities=correct)),
        (row("b"), prediction("b", "escalate", probabilities=wrong)),
        (row("c"), prediction("c")),
    ]
    result = metric_summary(joined)
    assert result["calibration_count"] == 2
    assert result["ece_10_bins"] == 0.5
    assert result["brier_score"] == 1.0
    unavailable = metric_summary([(row(), prediction())])
    assert unavailable["ece_10_bins"] is None
    assert unavailable["brier_score"] is None
    assert unavailable["input_tokens_sum"] is None


def test_malformed_probabilities_do_not_make_false_calibration():
    for probs in ({"refund": 0.9}, {a: 1.0 for a in ACTIONS}, {a: float("nan") for a in ACTIONS}):
        result = metric_summary([(row(), prediction(probabilities=probs))])
        assert result["calibration_count"] == 0


def test_generated_masked_slice_separate_from_actual_observations():
    joined = [
        (row("a"), prediction("a")),
        (row("b", boundary="missing_observation"), prediction("b")),
        (row("c", boundary="after_completed_action"), prediction("c")),
    ]
    assert len(slice_rows(joined, "observed")) == 2
    assert len(slice_rows(joined, "inspection")) == 1
    assert len(slice_rows(joined, "missing_observation")) == 1


def test_family_bootstrap_pairs_only_common_ids():
    data = [row("a", group="one"), row("b", group="two")]
    first = [(r, prediction(r["example_id"])) for r in data]
    second = [(r, prediction(r["example_id"], "defer")) for r in data]
    result = paired_bootstrap(first, second, samples=100)
    assert result["families"] == 2
    assert result["paired_rows"] == 2
    assert result["accuracy_difference"] == result["ci95_low"] == result["ci95_high"] == 1.0
    assert result == paired_bootstrap(first, second, samples=100)
    one_group = paired_bootstrap(first, second[:1])
    assert one_group["families"] == 1
    assert one_group["ci95_low"] is None


def test_prohibited_proposal_preserved_even_when_inference_error():
    r = {**row(action="escalate"), "prohibited_actions": ["refund"]}
    result = metric_summary([(r, prediction(error="blocked"))])
    assert result["prohibited_proposals"] == 1
    assert result["accuracy"] == 0


def test_report_writes_finite_machine_readable_metrics(tmp_path):
    data, preds = tmp_path / "data.jsonl", tmp_path / "predictions.jsonl"
    data.write_text(json.dumps(row()) + "\n")
    preds.write_text(json.dumps(prediction()) + "\n")
    result = generate_report(
        data,
        {"llm": preds},
        tmp_path / "report",
        plots=False,
        latency_scopes={"llm": "Full request"},
    )
    assert result["arms"]["llm"]["latency_scope"] == "Full request"
    assert result["arms"]["llm"]["slices"]["all"]["accuracy"] == 1
    assert (tmp_path / "report" / "metrics.csv").exists()
    assert json.loads((tmp_path / "report" / "metrics.json").read_text()) == result
