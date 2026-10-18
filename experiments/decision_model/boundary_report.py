"""Offline report of paired authorization-boundary repairs on exposed families."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.decision_model.next_evaluate import digest, task_outcomes, write
from experiments.decision_model.next_report import arm_metrics

ARMS = ("released_old", "released_new", "llm_old", "llm_new")
POLICIES = ("released", "llm")
LABELS = {
    "released_old": "Released + previous runtime",
    "released_new": "Released + repaired runtime",
    "llm_old": "LLM-only + previous runtime",
    "llm_new": "LLM-only + repaired runtime",
}


def analyze(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("complete") is not True or manifest.get("error"):
        raise ValueError("Boundary report requires a completed bounded evaluation")
    if manifest.get("cohort_previously_exposed") is not True:
        raise ValueError("Boundary report requires an explicitly exposed frozen cohort")
    if manifest.get("repeats") != 2:
        raise ValueError("Paired design requires exactly two attempts")
    cohort = json.loads((run / "cohort-manifest.json").read_text())
    if digest(run / "cohort-manifest.json") != manifest["cohort_manifest_sha256"]:
        raise ValueError("Saved cohort differs from final manifest")
    tasks = cohort["tasks"]
    if not isinstance(tasks, dict):
        tasks = {task["task_id"]: task for task in tasks}
    selected = manifest["task_ids"]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Final task identities must be nonempty and unique")
    if any(task not in tasks or tasks[task]["split"] != "test" for task in selected):
        raise ValueError("Final tasks must belong to the frozen test split")
    expected = {(arm, task, attempt) for arm in ARMS for task in selected for attempt in range(2)}
    if manifest.get("planned_trials") != len(expected):
        raise ValueError("Manifest planned coverage differs from paired design")
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines() if line]
    indexed = {}
    for row in rows:
        key = (row["arm"], row["task_id"], row["attempt"])
        if key in indexed or key not in expected:
            raise ValueError("Duplicate or unexpected arm/task/attempt result")
        recalculated = task_outcomes(row, tasks[row["task_id"]])
        for field, value in recalculated.items():
            if row.get(field) != value:
                raise ValueError(f"Outcome differs from native/tool evidence: {key}/{field}")
        if row["completed_and_verified"] and (
            row.get("tool_evidence_complete") is False or row.get("indeterminate_tool_calls")
        ):
            raise ValueError("Incomplete tool evidence cannot establish successful completion")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Missing paired final coverage")
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    families: dict[str, list[str]] = defaultdict(list)
    for task in selected:
        families[tasks[task]["composition_root"]].append(task)
    pairs = []
    comparisons = {}
    for policy in POLICIES:
        policy_pairs = []
        for task in selected:
            for attempt in range(2):
                old = indexed[(policy + "_old", task, attempt)]
                new = indexed[(policy + "_new", task, attempt)]
                pair = {
                    "policy": policy,
                    "task_id": task,
                    "composition_root": tasks[task]["composition_root"],
                    "attempt": attempt,
                    "completion_delta": int(new["completed_and_verified"])
                    - int(old["completed_and_verified"]),
                }
                for version, row in (("old", old), ("new", new)):
                    for field in (
                        "completed_and_verified",
                        "successful_unwanted_writes",
                        "missing_required_operations",
                        "successful_required_operations",
                        "llm_calls",
                        "authorization_blocked_attempts",
                        "invoice_scope_blocked_attempts",
                    ):
                        pair[version + "_" + field] = row.get(field)
                policy_pairs.append(pair)
        pairs.extend(policy_pairs)
        comparisons[policy] = {
            "improved_trial_pairs": sum(pair["completion_delta"] == 1 for pair in policy_pairs),
            "regressed_trial_pairs": sum(pair["completion_delta"] == -1 for pair in policy_pairs),
            "same_trial_pairs": sum(pair["completion_delta"] == 0 for pair in policy_pairs),
            "paired_trials": len(policy_pairs),
            "per_family_completion_delta": {
                family: sum(
                    pair["completion_delta"]
                    for pair in policy_pairs
                    if pair["composition_root"] == family
                )
                for family in families
            },
        }
    return {
        "schema_version": "exposed-paired-boundary-report-v1",
        "unique_tasks": len(selected),
        "composition_roots": len(families),
        "attempts_per_arm_task": 2,
        "total_trials": len(rows),
        "cohort_previously_exposed": True,
        "arms": {arm: arm_metrics(group) for arm, group in by_arm.items()},
        "per_family": {
            family: {
                arm: arm_metrics([row for row in by_arm[arm] if row["task_id"] in identifiers])
                for arm in ARMS
            }
            for family, identifiers in families.items()
        },
        "paired_trials": pairs,
        "new_vs_old": comparisons,
        "execution": {
            key: manifest.get(key)
            for key in (
                "arm_contracts",
                "thresholds",
                "device",
                "model_load_ms",
                "fallback_load_ms",
                "run_total_wall_ms",
                "invoice_local_fallback",
                "max_tool_calls",
                "max_llm_calls",
                "fallback_timeout",
                "max_job_seconds",
                "freeze_sha256",
            )
        },
        "limitations": [
            "These families were exposed by the previous experiment and informed the present "
            "repairs. This is an engineering regression, not new held-out evidence.",
            "Two variants per family and two attempts per task are correlated observations; "
            "the family count, not the trial count, describes independent authored structures.",
            "Both arms use frozen released model weights and existing calibration. No training, "
            "checkpoint selection, or calibration fit occurs in this comparison. A difference "
            "measures the combined boundary repairs, not learned model improvement.",
            "Request parsing, no-write completion, and tool descriptions change together; "
            "this design does not identify their separate causal contributions.",
            "Matched trials share tasks, underlying billing service, and budgets. Authorization "
            "parsing differs by runtime, so model inputs can differ even before the first write. "
            "Later histories depend on actions. Provider calls can vary at temperature zero.",
            "Reusing the earlier routing threshold does not validate calibration for the new "
            "runtime. Fewer calls can indicate omitted work; read usage alongside completion.",
            "Policy latency excludes Docker setup, native verification, and cleanup. Small-sample "
            "p95 is descriptive. Token totals contain returned usage and can omit failed calls; "
            "no dollar cost is inferred.",
            "Indeterminate tool effects are retained separately. Zero observed unwanted writes "
            "is not proof of zero harm when tool evidence is incomplete.",
        ],
        "provenance": {
            "manifest_sha256": digest(run / "manifest.json"),
            "results_sha256": digest(run / "results.jsonl"),
            "cohort_manifest_sha256": digest(run / "cohort-manifest.json"),
            "reporter_sha256": digest(Path(__file__)),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Exposed-family paired authorization-boundary evaluation",
        "",
        f"{report['unique_tasks']} previously exposed tasks across "
        f"{report['composition_roots']} composition "
        f"roots, two attempts per arm and task ({report['total_trials']} native trials).",
        "",
        "Weights and calibration are frozen. This measures authorization and runtime repairs "
        "before "
        "further training; it does not demonstrate a learned improvement.",
        "",
        "| Arm | Completed + verified | Native passes | Required writes | Unwanted writes | "
        "LLM calls | Both attempts complete |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        value = report["arms"][arm]
        lines.append(
            f"| {LABELS[arm]} | {value['completed_and_verified']}/{value['trials']} | "
            f"{value['verifier_passes']}/{value['trials']} | "
            f"{value['successful_required_operations']}/{value['required_operations']} | "
            f"{value['successful_unwanted_writes']} | {value['llm_calls']} | "
            f"{value['tasks_completed_both_attempts']}/{value['unique_tasks']} |"
        )
    lines += [
        "",
        "Completion requires policy completion and the independent native verifier. "
        "Withheld required operations, unresolved work, and blocked writes are not successes.",
        "",
        "| Policy | Improved pairs | Regressed pairs | Unchanged pairs |",
        "|---|---:|---:|---:|",
    ]
    for policy, value in report["new_vs_old"].items():
        lines.append(
            f"| {policy} | {value['improved_trial_pairs']} | "
            f"{value['regressed_trial_pairs']} | {value['same_trial_pairs']} |"
        )
    lines += [
        "",
        "| Composition root | Released old | Released repaired | LLM old | LLM repaired |",
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
    lines += [
        "",
        "| Arm | Policy p50 ms | Policy p95 ms | Unresolved invoices | "
        "Incomplete evidence trials |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        value = report["arms"][arm]
        latency = value["end_to_end_policy_ms"]

        def number(value: float | None) -> str:
            return "unavailable" if value is None else f"{value:.2f}"

        lines.append(
            f"| {LABELS[arm]} | {number(latency['p50'])} | {number(latency['p95'])} | "
            f"{value['unresolved_invoice_count']} | "
            f"{value['incomplete_tool_evidence_tasks']} |"
        )
    lines += [""] + ["- " + limitation for limitation in report["limitations"]]
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
