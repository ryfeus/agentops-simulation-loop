"""Offline report of released-decider versus LLM routing on fresh frozen families."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.decision_model.next_evaluate import digest, task_outcomes, write
from experiments.decision_model.next_report import arm_metrics

ARMS = ("released", "llm")
LABELS = {"released": "Released decider + fallback", "llm": "LLM-only"}


def analyze(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("complete") is not True or manifest.get("error"):
        raise ValueError("Generalization report requires a completed bounded evaluation")
    if manifest.get("cohort_previously_exposed") is not False:
        raise ValueError("Generalization report requires an explicitly unexposed frozen cohort")
    if manifest.get("repeats") != 2:
        raise ValueError("Paired design requires exactly two attempts")
    cohort = json.loads((run / "cohort-manifest.json").read_text())
    if (
        cohort.get("cohort_previously_exposed") is not False
        or cohort.get("pre_freeze_policy_rollouts") != 0
        or cohort.get("pre_freeze_model_inferences") != 0
    ):
        raise ValueError("Cohort does not support prospective evaluation claim")
    if digest(run / "cohort-manifest.json") != manifest["cohort_manifest_sha256"]:
        raise ValueError("Saved cohort differs from final manifest")
    access = run / "evaluation-access.json"
    access_record = json.loads(access.read_text())
    if (
        digest(access) != manifest.get("evaluation_access_sha256")
        or access_record.get("cohort_manifest_sha256") != manifest["cohort_manifest_sha256"]
        or access_record.get("freeze_sha256") != manifest.get("freeze_sha256")
        or access_record.get("output") != str(run.resolve())
        or access_record.get("claimed_before_model_initialization") is not True
    ):
        raise ValueError("Prospective access claim does not bind this run")
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
    for task in selected:
        for attempt in range(2):
            released = indexed[("released", task, attempt)]
            llm = indexed[("llm", task, attempt)]
            pair = {
                "task_id": task,
                "composition_root": tasks[task]["composition_root"],
                "attempt": attempt,
                "completion_delta": int(released["completed_and_verified"])
                - int(llm["completed_and_verified"]),
            }
            for arm, row in (("released", released), ("llm", llm)):
                for field in (
                    "completed_and_verified",
                    "verifier_pass",
                    "successful_unwanted_writes",
                    "missing_required_operations",
                    "successful_required_operations",
                    "llm_calls",
                    "llm_input_tokens",
                    "llm_output_tokens",
                    "end_to_end_policy_ms",
                    "authorization_blocked_attempts",
                    "invoice_scope_blocked_attempts",
                ):
                    pair[arm + "_" + field] = row.get(field)
            pairs.append(pair)
    comparisons = {
        "improved_trial_pairs": sum(pair["completion_delta"] == 1 for pair in pairs),
        "regressed_trial_pairs": sum(pair["completion_delta"] == -1 for pair in pairs),
        "same_trial_pairs": sum(pair["completion_delta"] == 0 for pair in pairs),
        "paired_trials": len(pairs),
        "per_family_completion_delta": {
            family: sum(
                pair["completion_delta"] for pair in pairs if pair["composition_root"] == family
            )
            for family in families
        },
    }
    return {
        "schema_version": "prospective-generalization-report-v1",
        "unique_tasks": len(selected),
        "composition_roots": len(families),
        "attempts_per_arm_task": 2,
        "total_trials": len(rows),
        "cohort_previously_exposed": False,
        "arms": {arm: arm_metrics(group) for arm, group in by_arm.items()},
        "per_family": {
            family: {
                arm: arm_metrics([row for row in by_arm[arm] if row["task_id"] in identifiers])
                for arm in ARMS
            }
            for family, identifiers in families.items()
        },
        "paired_trials": pairs,
        "released_vs_llm": comparisons,
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
                "evaluation_access_sha256",
            )
        },
        "limitations": [
            "Fresh authored compositions share known billing primitives and renderer; this is "
            "not an external benchmark or broad natural-language generalization claim.",
            "Two variants per family and two attempts per task are correlated. Six roots, "
            "not 48 trials, describe the authored structural coverage.",
            "Both arms use the same repaired runtime, guards and initial observations. "
            "Only the released arm routes through the fixed decision model; later histories "
            "depend on actual actions. No weights, thresholds or calibration were fitted.",
            "The comparison measures routing with a released model, not learned improvement. "
            "The reused 0.7 threshold is not a demonstrated calibration guarantee.",
            "Provider responses may vary at temperature zero. Fewer calls can indicate omitted "
            "work, so compare calls with completion, safety and missing required operations.",
            "Native completion does not score all clarification or reporting prose. "
            "Harness target discovery is supplied and not evaluated.",
            "Policy latency excludes Docker setup, verification and cleanup. Model startup is "
            "separate; small-sample p95 is descriptive. Returned tokens can omit failed usage; "
            "no dollar cost is inferred.",
            "Missing or indeterminate tool evidence is reported separately; zero observed "
            "unwanted writes alone does not prove safety beyond this cohort.",
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
        "# Prospective released-decider versus LLM evaluation",
        "",
        f"{report['unique_tasks']} fresh authored tasks across "
        f"{report['composition_roots']} roots; "
        f"two attempts per arm and task ({report['total_trials']} native trials).",
        "",
        "The same repaired runtime is used in both arms. Model weights and calibration remain "
        "frozen; this tests routing on new families, not learned improvement.",
        "",
        "| Arm | Strict completion | Native passes | Required writes | Unwanted writes | "
        "Both attempts complete |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        value = report["arms"][arm]
        lines.append(
            f"| {LABELS[arm]} | {value['completed_and_verified']}/{value['trials']} | "
            f"{value['verifier_passes']}/{value['trials']} | "
            f"{value['successful_required_operations']}/{value['required_operations']} | "
            f"{value['successful_unwanted_writes']} | "
            f"{value['tasks_completed_both_attempts']}/{value['unique_tasks']} |"
        )
    pair = report["released_vs_llm"]
    lines += [
        "",
        "Released decider versus LLM on matched task-attempt pairs: "
        f"{pair['improved_trial_pairs']} wins, {pair['regressed_trial_pairs']} losses, "
        f"{pair['same_trial_pairs']} ties.",
        "",
        "| Composition root | Released decider | LLM-only |",
        "|---|---:|---:|",
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
        "| Arm | LLM calls | Returned input tokens | Returned output tokens | "
        "Policy p50 ms | Policy p95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        value = report["arms"][arm]
        latency = value["end_to_end_policy_ms"]

        def number(value: float | None) -> str:
            return "unavailable" if value is None else f"{value:.2f}"

        lines.append(
            f"| {LABELS[arm]} | {value['llm_calls']} | {value['llm_input_tokens']} | "
            f"{value['llm_output_tokens']} | {number(latency['p50'])} | "
            f"{number(latency['p95'])} |"
        )
    lines += [
        "",
        "Strict completion requires policy resolution and the native verifier. "
        "Failed trials remain in denominators and usage. Per-arm JSON metrics include "
        "missing work, blocked attempts, inference/execution errors and indeterminate tools.",
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
