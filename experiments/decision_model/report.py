"""Strict paired decision replay metrics; plotting dependencies are isolated and optional."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.decision_model.data import ACTIONS

ERROR = "error"
SLICES = ("all", "observed", "inspection", "completed_action", "missing_observation")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def join_predictions(
    data: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    allow_partial: bool = False,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    """The dataset, not prediction metadata, defines gold, groups, and test membership."""
    known = {row["example_id"]: row for row in data}
    if len(known) != len(data):
        raise ValueError("Duplicate dataset example_id")
    expected = {key for key, row in known.items() if row["split"] == "test"}
    if not expected:
        raise ValueError("Dataset contains no test rows")
    found: dict[str, dict[str, Any]] = {}
    filtered = 0
    for prediction in predictions:
        key = prediction["example_id"]
        if key not in known:
            raise ValueError(f"Unknown prediction example_id: {key}")
        if key not in expected:
            filtered += 1
            continue
        if key in found:
            raise ValueError(f"Duplicate prediction example_id: {key}; repeats are not samples")
        found[key] = prediction
    missing = expected - found.keys()
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} of {len(expected)} test predictions")
    if not found:
        raise ValueError("Arm has no test predictions")
    joined = [(known[key], found[key]) for key in sorted(found)]
    return joined, {
        "expected": len(expected),
        "available": len(found),
        "missing": len(missing),
        "filtered_non_test_predictions": filtered,
    }


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _errored(prediction: dict[str, Any]) -> bool:
    return bool(prediction.get("error")) or prediction.get("action") not in ACTIONS


def _correct(row: dict[str, Any], prediction: dict[str, Any]) -> bool:
    return not _errored(prediction) and prediction["action"] == row["label_action"]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _probabilities(prediction: dict[str, Any]) -> dict[str, float] | None:
    probabilities = prediction.get("probabilities")
    if _errored(prediction) or not isinstance(probabilities, dict):
        return None
    if set(probabilities) != set(ACTIONS):
        return None
    if not all(_finite_number(p) and 0 <= p <= 1 for p in probabilities.values()):
        return None
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-4):
        return None
    return probabilities


def metric_summary(joined: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    n = len(joined)
    matrix = {action: dict.fromkeys((*ACTIONS, ERROR), 0) for action in ACTIONS}
    correct = errors = defer = prohibited = accepted = accepted_correct = 0
    latencies, input_tokens, output_tokens, calibration = [], [], [], []
    for row, prediction in joined:
        gold = row["label_action"]
        action = prediction.get("action")
        error = _errored(prediction)
        matrix[gold][ERROR if error else action] += 1
        correct += _correct(row, prediction)
        errors += error
        defer += action == "defer" and not error
        # Preserve even an errored model's proposed harmful action if one was recorded.
        prohibited += action in row.get("prohibited_actions", [])
        if not error and action != "defer":
            accepted += 1
            accepted_correct += _correct(row, prediction)
        latency = prediction.get("latency_ms")
        if _finite_number(latency) and latency >= 0:
            latencies.append(latency)
        for key, values in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
            value = prediction.get(key)
            if _finite_number(value) and value >= 0:
                values.append(value)
        probabilities = _probabilities(prediction)
        if probabilities is not None:
            confidence = probabilities[action]
            brier = sum((probabilities[a] - float(a == gold)) ** 2 for a in ACTIONS)
            calibration.append((confidence, int(_correct(row, prediction)), brier))
    per_class, f1s = {}, []
    for action in ACTIONS:
        true_positive = matrix[action][action]
        support = sum(matrix[action].values())
        predicted = sum(matrix[a][action] for a in ACTIONS)
        f1 = 2 * true_positive / (support + predicted) if support + predicted else None
        per_class[action] = {
            "support": support,
            "precision": true_positive / predicted if predicted else None,
            "recall": true_positive / support if support else None,
            "f1": f1,
        }
        # Macro F1 uses the declared four-class vocabulary, not a changing subset.
        f1s.append(f1 or 0.0)
    bins = []
    for i in range(10):
        bucket = [r for r in calibration if min(int(r[0] * 10), 9) == i]
        bins.append(
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "count": len(bucket),
                "accuracy": sum(r[1] for r in bucket) / len(bucket) if bucket else None,
                "confidence": sum(r[0] for r in bucket) / len(bucket) if bucket else None,
            }
        )
    ece = (
        (
            sum(b["count"] * abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"])
            / len(calibration)
        )
        if calibration
        else None
    )
    return {
        "rows": n,
        "tasks": len({r["task_id"] for r, _ in joined}),
        "families": len({r["group_id"] for r, _ in joined}),
        "correct": correct,
        "accuracy": correct / n if n else None,
        "macro_f1": sum(f1s) / len(ACTIONS) if n else None,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "prohibited_proposals": prohibited,
        "prohibited_rate": prohibited / n if n else None,
        "errors": errors,
        "error_rate": errors / n if n else None,
        "defer_count": defer,
        "defer_rate": defer / n if n else None,
        "fallback_needed_rate": (defer + errors) / n if n else None,
        "non_defer_count": accepted,
        "non_defer_coverage": accepted / n if n else None,
        "non_defer_accuracy": accepted_correct / accepted if accepted else None,
        "gold_defer_count": sum(r["label_action"] == "defer" for r, _ in joined),
        "latency_count": len(latencies),
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "input_tokens_count": len(input_tokens),
        "input_tokens_sum": sum(input_tokens) if input_tokens else None,
        "input_tokens_p50": percentile(input_tokens, 0.5),
        "input_tokens_p95": percentile(input_tokens, 0.95),
        "output_tokens_count": len(output_tokens),
        "output_tokens_sum": sum(output_tokens) if output_tokens else None,
        "calibration_count": len(calibration),
        "ece_10_bins": ece,
        "brier_score": sum(r[2] for r in calibration) / len(calibration) if calibration else None,
        "calibration_bins": bins,
    }


def slice_rows(
    joined: list[tuple[dict[str, Any], dict[str, Any]]],
    name: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if name == "all":
        return joined
    if name == "observed":
        return [(row, pred) for row, pred in joined if row["boundary"] != "missing_observation"]
    boundary = {
        "inspection": "after_inspection",
        "completed_action": "after_completed_action",
        "missing_observation": "missing_observation",
    }[name]
    return [(row, pred) for row, pred in joined if row["boundary"] == boundary]


def paired_bootstrap(
    first: list[tuple[dict[str, Any], dict[str, Any]]],
    second: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    seed: int = 17,
    samples: int = 2000,
) -> dict[str, Any]:
    """Paired cluster bootstrap: resample complete prototype groups, never individual rows."""
    second_by_id = {row["example_id"]: prediction for row, prediction in second}
    differences: dict[str, list[int]] = defaultdict(list)
    for row, prediction in first:
        if row["example_id"] in second_by_id:
            differences[row["group_id"]].append(
                int(_correct(row, prediction)) - int(_correct(row, second_by_id[row["example_id"]]))
            )
    groups = sorted(differences)
    count = sum(len(values) for values in differences.values())
    point = sum(sum(values) for values in differences.values()) / count if count else None
    estimates = []
    if len(groups) > 1:
        rng = random.Random(seed)
        for _ in range(samples):
            selected = rng.choices(groups, k=len(groups))
            estimates.append(
                sum(sum(differences[g]) for g in selected)
                / sum(len(differences[g]) for g in selected)
            )
    return {
        "paired_rows": count,
        "families": len(groups),
        "accuracy_difference": point,
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
        "bootstrap_samples": len(estimates),
        "seed": seed,
        "method": "Paired prototype cluster percentile bootstrap; row-weighted accuracy delta.",
        "limitation": "Few families: exploratory interval, not generalization proof.",
    }


def _mapping(values: list[str], label: str) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected {label} NAME=VALUE: {value}")
        name, item = value.split("=", 1)
        if not name or not item or name in result:
            raise ValueError(f"Empty or duplicate {label} name: {name}")
        result[name] = item
    return result


def write_plots(report: dict[str, Any], output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plots require matplotlib; run in an isolated environment with it installed "
            "or use --no-plots for metrics only"
        ) from error
    arms = report["arms"]
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (name, arm) in enumerate(arms.items()):
        values = arm["slices"]["observed"]
        latency, quality = values["latency_p50_ms"], values["accuracy"]
        if latency is None or quality is None:
            continue
        ax.scatter(max(latency, 1e-6), quality, s=80, color=colors(i % 10), label=name)
        ax.annotate(
            name,
            (max(latency, 1e-6), quality),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("Reported median latency, ms (log scale; timing boundaries differ)")
    ax.set_ylabel("Decision accuracy (errors count as incorrect)")
    ax.set_title("Observed tool snapshots: decision quality and measured latency")
    ax.grid(alpha=0.2)
    caption = "Missing-observation ablations excluded. No agent speedup inferred.\n" + "; ".join(
        f"{name}: {arm['latency_scope']}" for name, arm in arms.items()
    )
    import textwrap

    fig.text(0.02, 0.015, "\n".join(textwrap.wrap(caption, width=130)), fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(output / "quality-vs-latency.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4.5), squeeze=False)
    for ax, (name, arm) in zip(axes[0], arms.items(), strict=True):
        matrix = arm["slices"]["observed"]["confusion_matrix"]
        counts = [[matrix[gold][pred] for pred in (*ACTIONS, ERROR)] for gold in ACTIONS]
        normalized = [[count / max(sum(row), 1) for count in row] for row in counts]
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i, row in enumerate(counts):
            for j, count in enumerate(row):
                ax.text(
                    j,
                    i,
                    str(count),
                    ha="center",
                    va="center",
                    color="white" if normalized[i][j] > 0.5 else "black",
                )
        ax.set_xticks(range(5), (*ACTIONS, ERROR), rotation=35, ha="right")
        ax.set_yticks(range(4), ACTIONS)
        ax.set_xlabel("Proposed decision")
        ax.set_ylabel("Gold decision")
        ax.set_title(name)
    fig.suptitle("Observed snapshots: counts; shading normalized within each gold class")
    fig.tight_layout()
    fig.savefig(output / "confusion-matrices.png", dpi=180)
    plt.close(fig)


def generate_report(
    data_path: Path,
    paths: dict[str, Path],
    output: Path,
    *,
    allow_partial: bool = False,
    latency_scopes: dict[str, str] | None = None,
    plots: bool = True,
) -> dict[str, Any]:
    data = read_jsonl(data_path)
    if not paths:
        raise ValueError("At least one prediction arm is required")
    arms, joined_arms = {}, {}
    for name, path in paths.items():
        joined, coverage = join_predictions(data, read_jsonl(path), allow_partial=allow_partial)
        joined_arms[name] = joined
        arms[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "coverage": coverage,
            "latency_scope": (latency_scopes or {}).get(name, "unspecified prediction timing"),
            "slices": {
                slice_name: metric_summary(slice_rows(joined, slice_name)) for slice_name in SLICES
            },
        }
    paired = []
    for first, second in itertools.combinations(arms, 2):
        for name in ("all", "observed", "inspection"):
            paired.append(
                {
                    "first": first,
                    "second": second,
                    "slice": name,
                    **paired_bootstrap(
                        slice_rows(joined_arms[first], name), slice_rows(joined_arms[second], name)
                    ),
                }
            )
    test = [row for row in data if row["split"] == "test"]
    report = {
        "data_path": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "test_rows": len(test),
        "test_families": len({r["group_id"] for r in test}),
        "test_tasks": len({r["task_id"] for r in test}),
        "test_boundaries": dict(Counter(row["boundary"] for row in test)),
        "allow_partial": allow_partial,
        "arms": arms,
        "paired_differences": paired,
        "definitions": {
            "accuracy": "Correct action / all rows, including gold defer; errors count incorrect.",
            "defer": "Abstention, not task completion. Fallback-needed rate includes errors too.",
            "non_defer_accuracy": "Correct among successful non-defer decisions only.",
            "observed": "Actual inspection and post-action snapshots; masked rows excluded.",
            "macro_f1": "Mean over four action classes; absent class contributes zero.",
            "calibration": "Valid four-action probabilities only. Ten equal-width ECE bins. "
            "Multiclass Brier sums squared class errors.",
            "rule_probabilities": "One-hot deterministic scores; uncertainty uncalibrated.",
            "latency": "Arm-specific prediction timing. Small-sample p95 is descriptive. "
            "Initialization excluded when source excluded it.",
            "sampling": "One prediction per example; related snapshots are correlated. "
            "Paired intervals resample prototype families, not rows.",
        },
        "limitations": [
            "Replay-only metrics do not establish Harbor task success or actual LLM-call savings.",
            "One-seed exploratory experiment; sparse family counts and calibration samples.",
            "Historical training families form a new diagnostic; not an out-of-distribution test.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    flat = []
    for arm, value in arms.items():
        for name, metrics in value["slices"].items():
            flat.append(
                {
                    "arm": arm,
                    "slice": name,
                    "latency_scope": value["latency_scope"],
                    **{
                        key: val
                        for key, val in metrics.items()
                        if not isinstance(val, (dict, list))
                    },
                }
            )
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    if plots:
        write_plots(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", action="append", required=True, metavar="NAME=FILE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--latency-scope", action="append", default=[], metavar="NAME=DESCRIPTION")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    paths = {name: Path(path) for name, path in _mapping(args.predictions, "predictions").items()}
    scopes = _mapping(args.latency_scope, "latency-scope")
    if scopes.keys() - paths.keys():
        parser.error("latency-scope references an unknown arm")
    report = generate_report(
        args.data,
        paths,
        args.output,
        allow_partial=args.allow_partial,
        latency_scopes=scopes,
        plots=not args.no_plots,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "arms": list(report["arms"]),
                "test_rows": report["test_rows"],
                "test_families": report["test_families"],
            }
        )
    )


if __name__ == "__main__":
    main()
