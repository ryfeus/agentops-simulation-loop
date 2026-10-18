"""Report a completed frozen four-arm native evaluation without running models/tools."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.decision_model.next_evaluate import digest, summarize, task_outcomes, write

ARMS = ("rule", "released", "tuned", "llm")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    left, right = math.floor(index), math.ceil(index)
    return values[left] + (values[right] - values[left]) * (index - left)


def arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = summarize(rows)
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tasks[row["task_id"]].append(row)
    result.update(
        trials=len(rows),
        unique_tasks=len(tasks),
        tasks_completed_both_attempts=sum(
            all(r["completed_and_verified"] for r in group) for group in tasks.values()
        ),
        tasks_completed_any_attempt=sum(
            any(r["completed_and_verified"] for r in group) for group in tasks.values()
        ),
    )
    for field in ("end_to_end_policy_ms", "total_wall_ms"):
        values = [
            row[field]
            for row in rows
            if isinstance(row.get(field), (int, float)) and math.isfinite(row[field])
        ]
        result[field] = {
            "observed_trials": len(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "sum": sum(values) if len(values) == len(rows) else None,
        }
    return result


def analyze(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("complete") is not True or manifest.get("error"):
        raise ValueError("Final report requires a completed bounded run, not partial outcomes")
    cohort = json.loads((run / "cohort-manifest.json").read_text())
    if digest(run / "cohort-manifest.json") != manifest["cohort_manifest_sha256"]:
        raise ValueError("Saved cohort differs from final manifest")
    tasks = cohort["tasks"]
    if not isinstance(tasks, dict):
        tasks = {task["task_id"]: task for task in tasks}
    selected = manifest["task_ids"]
    if len(selected) != len(set(selected)) or any(
        tasks[task]["split"] != "test" for task in selected
    ):
        raise ValueError("Final task identities are duplicated or not test split")
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines() if line]
    expected = {(arm, task, attempt) for arm in ARMS for task in selected for attempt in range(2)}
    indexed = {}
    for row in rows:
        key = (row["arm"], row["task_id"], row["attempt"])
        if key in indexed:
            raise ValueError("Duplicate arm/task/attempt result")
        if key not in expected:
            raise ValueError("Unexpected final result identity")
        recalculated = task_outcomes(row, tasks[row["task_id"]])
        for field in (
            "completed_and_verified",
            "verifier_pass",
            "successful_unwanted_writes",
            "missing_required_operations",
            "successful_required_operations",
            "required_operations",
        ):
            if row.get(field) != recalculated[field]:
                raise ValueError(
                    f"Outcome differs from native verifier/tool evidence: {key}/{field}"
                )
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Missing paired final coverage")
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    families: dict[str, list[str]] = defaultdict(list)
    for task in selected:
        families[tasks[task]["composition_root"]].append(task)
    pairs = []
    for task in selected:
        for attempt in range(2):
            pair = {
                "task_id": task,
                "composition_root": tasks[task]["composition_root"],
                "attempt": attempt,
            }
            for arm in ARMS:
                row = indexed[(arm, task, attempt)]
                for field in (
                    "completed_and_verified",
                    "successful_unwanted_writes",
                    "missing_required_operations",
                    "llm_calls",
                    "authorization_blocked_attempts",
                ):
                    pair[arm + "_" + field] = row.get(field)
            pair["tuned_minus_released_completion"] = int(
                indexed[("tuned", task, attempt)]["completed_and_verified"]
            ) - int(indexed[("released", task, attempt)]["completed_and_verified"])
            pairs.append(pair)
    return {
        "schema_version": "whole-task-finetune-four-arm-report-v1",
        "unique_tasks": len(selected),
        "composition_roots": len(families),
        "attempts_per_arm_task": 2,
        "cohort_previously_exposed": manifest.get("cohort_previously_exposed", False),
        "selected_checkpoint_role": manifest.get("selected_checkpoint_role", "unspecified"),
        "total_trials": len(rows),
        "arms": {arm: arm_metrics(group) for arm, group in by_arm.items()},
        "per_family": {
            family: {
                arm: arm_metrics([row for row in by_arm[arm] if row["task_id"] in identifiers])
                for arm in ARMS
            }
            for family, identifiers in families.items()
        },
        "paired_trials": pairs,
        "tuned_vs_released": {
            "improved_trial_pairs": sum(
                row["tuned_minus_released_completion"] == 1 for row in pairs
            ),
            "regressed_trial_pairs": sum(
                row["tuned_minus_released_completion"] == -1 for row in pairs
            ),
            "same_trial_pairs": sum(row["tuned_minus_released_completion"] == 0 for row in pairs),
        },
        "execution": {
            key: manifest.get(key)
            for key in (
                "thresholds",
                "device",
                "model_load_ms",
                "fallback_load_ms",
                "run_total_wall_ms",
                "escalation_history",
                "invoice_authorization",
                "invoice_local_fallback",
                "max_tool_calls",
                "max_llm_calls",
                "fallback_timeout",
                "max_job_seconds",
            )
        },
        "limitations": [
            (
                "This is a previously exposed cohort regression, not new holdout evidence. "
                "Earlier outcomes informed follow-up policy changes."
                if manifest.get("cohort_previously_exposed", False)
                else "The final cohort holds out fresh authored compositions, "
                "not a new billing domain or external benchmark."
            ),
            "Two attempts per task are correlated repetitions; "
            "family and distinct-task outcomes are reported separately.",
            "Development selected standalone autonomous completion without LLM fallback; "
            "final results include the existing LLM continuation.",
            "Temperature and routing thresholds use reserved calibration examples/tasks, "
            "separate from development and final test. Reusing those values after a routing "
            "change does not establish calibration for the new routing policy.",
            "All arms use the same observed history, invoice authorization guard, "
            "initial state, and tool budget; later states depend on actual actions.",
            "Invoice-local mode adds a shared 16-model-call cap per task; legacy whole-task "
            "routing used recursion and tool bounds without this explicit shared cap. "
            "A comparison with prior runs therefore changes routing and budget enforcement.",
            "Fewer LLM calls can reflect omissions or refusals; completed tasks "
            "and missing required operations must be considered together.",
            "Policy latency excludes Docker setup/verifier/cleanup; total task wall includes them. "
            "Process time includes model startup and all four arms.",
            "Token totals contain returned provider usage; failures can omit usage. "
            "No dollar costs are inferred.",
            "Cancellation preserves completed tool evidence and known token usage. "
            "If a tool was in flight at cancellation, its effect is uncertain; reported "
            "successful writes are lower bounds for incomplete_tool_evidence_tasks, "
            "not proof of zero harm. Native final-state verification is retained separately.",
        ],
        "provenance": {
            "manifest_sha256": digest(run / "manifest.json"),
            "results_sha256": digest(run / "results.jsonl"),
            "cohort_sha256": digest(run / "cohort-manifest.json"),
            "reporter_sha256": digest(Path(__file__)),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    exposed = report.get("cohort_previously_exposed", False)
    selected_baseline = report.get("selected_checkpoint_role") == "released_baseline"
    labels = {arm: arm for arm in ARMS}
    if selected_baseline:
        labels["tuned"] = "selected released baseline"
    lines = [
        "# Whole-task decision-policy comparison",
        "",
        f"{report['unique_tasks']} "
        + ("previously exposed cohort tasks" if exposed else "held-out tasks")
        + f" across {report['composition_roots']} composition roots, "
        "two attempts per task and arm.",
        "",
    ]
    if selected_baseline:
        lines += [
            "The raw `tuned` arm is the selected epoch-0 released baseline. "
            "This comparison does not demonstrate a learned improvement.",
            "",
        ]
    lines += [
        "| Arm | Completed + verified trials | Unwanted writes | Missing required operations "
        "| LLM calls | Both attempts complete |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, metrics in report["arms"].items():
        lines.append(
            f"| {labels[arm]} | {metrics['completed_and_verified']}/{metrics['trials']} | "
            f"{metrics['successful_unwanted_writes']} | {metrics['missing_required_operations']} | "
            f"{metrics['llm_calls']} | "
            f"{metrics['tasks_completed_both_attempts']}/{metrics['unique_tasks']} |"
        )
    lines += [
        "",
        "Each completion requires explicit policy completion "
        "and the native state/trajectory verifier. "
        "Authorization blocks and withheld required work are not successes.",
        "",
        "| Composition root | Rule | Released | "
        + ("Selected baseline" if selected_baseline else "Tuned")
        + " | LLM |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, arms in report["per_family"].items():
        lines.append(
            "| "
            + family
            + " | "
            + " | ".join(
                f"{arms[arm]['completed_and_verified']}/{arms[arm]['trials']}" for arm in ARMS
            )
            + " |"
        )
    pair = report["tuned_vs_released"]
    lines += [
        "",
        ("Selected baseline" if selected_baseline else "Tuned")
        + " versus released on matched task-attempt pairs: "
        f"{pair['improved_trial_pairs']} improved, "
        f"{pair['regressed_trial_pairs']} regressed, {pair['same_trial_pairs']} unchanged.",
        "",
    ]
    lines += ["- " + limitation for limitation in report["limitations"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.run)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "metrics.json", report)
    (args.output / "results.md").write_text(markdown(report))
    with (args.output / "paired-trials.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["paired_trials"][0]))
        writer.writeheader()
        writer.writerows(report["paired_trials"])
    print(
        json.dumps({arm: value["completed_and_verified"] for arm, value in report["arms"].items()})
    )


if __name__ == "__main__":
    main()
