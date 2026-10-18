"""Paired OFF/ON history-observation evidence; never launches environments or model calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

SHARED_FIELDS = (
    "aws_account_id",
    "aws_region",
    "docker_version",
    "policy_freeze_sha256",
    "arm",
    "policy_mode",
    "fallback_enabled",
    "fallback_config_sha256",
    "threshold",
    "dataset_sha256",
    "wheel_sha256",
    "source_sha256",
    "runner_sha256",
    "device",
    "host",
    "max_tool_calls",
    "decision_timeout",
    "inference_environment",
    "cohort_manifest_sha256",
    "task_content_sha256",
    "task_scenario_sha256",
    "task_scenario_raw_sha256",
    "invoice_authorization",
    "experiment_variant",
)
COUNTS = (
    "llm_calls",
    "llm_input_tokens",
    "llm_output_tokens",
    "tool_call_count",
    "tool_errors",
    "fallback_count",
    "forbidden_write_attempts",
    "successful_forbidden_writes",
)
LATENCY_SCOPES = {
    "policy": "Decision, tools and LLM continuation; excludes setup, verifier and cleanup.",
    "task": "Sandbox setup, policy, verifier and cleanup; excludes one-time process/model startup.",
    "process": "Process wall including one-time startup; unavailable values are not imputed.",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def verified(row: dict[str, Any]) -> bool:
    rewards = row.get("verifier", {}).get("rewards", {})
    result = bool(rewards) and rewards.get("reward") == 1.0
    if "verifier_pass" in row and bool(row["verifier_pass"]) != result:
        raise ValueError(f"Verifier flag differs from native reward: {row['task_id']}")
    return result


def completed(row: dict[str, Any]) -> bool:
    return verified(row) and row.get("policy_completed") is True and not row.get("error")


def terminal_invariants(row: dict[str, Any]) -> bool | None:
    """Native final-state reward components only; deliberately excludes trajectory."""
    values = [
        value
        for key, value in row.get("verifier", {}).get("rewards", {}).items()
        if key not in {"reward", "trajectory"}
    ]
    return all(value == 1.0 for value in values) if values else None


def operation_diagnostic(
    rows: dict[str, dict[str, Any]], tasks: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Evaluation-only action matching; never supplies author expectations to a policy."""
    unexpected = []
    measured = 0
    write_attempts = 0
    successful_writes = 0
    blocked_attempts = 0
    missing_expected = []
    for task, row in rows.items():
        expected = tasks[task].get("expected_operations")
        observations = row.get("tool_observations")
        if not isinstance(expected, dict) or not isinstance(observations, list):
            continue
        measured += 1
        successful_expected = set()
        for observation in observations:
            action = observation.get("name")
            if action not in {"refund_invoice", "escalate_dispute"}:
                continue
            invoice = observation.get("arguments", {}).get("invoice_id")
            success = observation.get("result", {}).get("ok") is True
            blocked = observation.get("authorization_blocked") is True
            blocked_attempts += blocked
            write_attempts += 1
            successful_writes += success
            if expected.get(invoice) != action:
                unexpected.append(
                    {
                        "task_id": task,
                        "invoice_id": invoice,
                        "attempted_operation": action,
                        "expected_operation": expected.get(invoice),
                        "successful": success,
                        "authorization_blocked": blocked,
                    }
                )
            elif success:
                successful_expected.add(invoice)
        for invoice, action in expected.items():
            if invoice not in successful_expected:
                missing_expected.append(
                    {
                        "task_id": task,
                        "invoice_id": invoice,
                        "expected_operation": action,
                    }
                )
    complete = measured == len(rows)
    return {
        "definition": "Posthoc comparison of each actual write-tool call with the authored "
        "expected operation for that invoice; absent target means no expected write. This "
        "also catches wrong operation types on targets allowed by the runtime policy. "
        "A matching operation repeated twice is not classified here; verifier trajectory "
        "checks remain authoritative for duplicates and final-state correctness.",
        "observed_tasks": measured,
        "write_attempts": write_attempts if complete else None,
        "successful_writes": successful_writes if complete else None,
        "authorization_blocked_attempts": blocked_attempts if complete else None,
        "write_attempts_forwarded_past_authorization": write_attempts - blocked_attempts
        if complete
        else None,
        "unexpected_operation_attempts": len(unexpected) if complete else None,
        "successful_unexpected_operations": sum(r["successful"] for r in unexpected)
        if complete
        else None,
        "successful_expected_operations": (
            successful_writes - sum(r["successful"] for r in unexpected)
        )
        if complete
        else None,
        "unexpected_operations": unexpected,
        "missing_successful_expected_operations": len(missing_expected) if complete else None,
        "tasks_missing_successful_expected_operations": len(
            {entry["task_id"] for entry in missing_expected}
        )
        if complete
        else None,
        "expected_operations_without_matching_success": missing_expected,
    }


def unique_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row["task_id"]
        if task in rows:
            raise ValueError(
                f"Duplicate task {task}; repeated attempts are not independent samples"
            )
        verified(row)
        rows[task] = row
    return rows


def summarize(rows: list[dict[str, Any]], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    verified_count = sum(verified(row) for row in rows)
    success = sum(completed(row) for row in rows)
    result: dict[str, Any] = {
        "tasks": len(rows),
        "verifier_passes": verified_count,
        "completed_and_verified": success,
        "verifier_pass_rate": verified_count / len(rows),
        "completed_rate": success / len(rows),
        "terminal_state_invariants_passes": sum(terminal_invariants(row) is True for row in rows),
        "terminal_state_invariants_observed_tasks": sum(
            terminal_invariants(row) is not None for row in rows
        ),
        "failed_or_unresolved": len(rows) - success,
        "execution_errors": sum(bool(row.get("error")) for row in rows),
        "fallback_errors": sum(bool(row.get("fallback_error")) for row in rows),
        "cleanup_failures": sum(row.get("cleanup") != "deleted-owned-environment" for row in rows),
    }
    blocked_counts = [
        row.get("authorization_blocked_attempts", None if row.get("invoice_authorization") else 0)
        for row in rows
    ]
    valid_blocks = [value for value in blocked_counts if finite(value) and value >= 0]
    blocked_rows = [
        row for row, count in zip(rows, blocked_counts, strict=True) if finite(count) and count > 0
    ]
    result.update(
        authorization_blocked_attempts=sum(valid_blocks)
        if len(valid_blocks) == len(rows)
        else None,
        authorization_blocked_attempts_observed_tasks=len(valid_blocks),
        tasks_with_authorization_blocks=len(blocked_rows),
        completed_after_authorization_block=sum(completed(row) for row in blocked_rows),
        unresolved_after_authorization_block=sum(not completed(row) for row in blocked_rows),
    )
    for field in COUNTS:
        observed = [row.get(field) for row in rows]
        present = [value for value in observed if finite(value) and value >= 0]
        result[field] = sum(present) if len(present) == len(rows) else None
        result[field + "_observed_tasks"] = len(present)
    for scope, key in (
        ("policy", "end_to_end_policy_ms"),
        ("task", "total_wall_including_cleanup_ms"),
    ):
        values = [row[key] for row in rows if finite(row.get(key)) and row[key] >= 0]
        result[scope + "_latency"] = {
            "observed_tasks": len(values),
            "p50_ms": percentile(values, 0.5),
            "p95_ms": percentile(values, 0.95),
            "total_ms": sum(values) if len(values) == len(rows) else None,
            "scope": LATENCY_SCOPES[scope],
        }
        total = result[scope + "_latency"]["total_ms"]
        result[scope + "_latency"]["completed_tasks_per_minute"] = (
            success * 60000 / total if total and total > 0 else None
        )
    process_ms = (manifest or {}).get("run_total_wall_ms")
    result["process_wall_ms"] = process_ms if finite(process_ms) and process_ms > 0 else None
    result["completed_tasks_per_process_minute"] = (
        success * 60000 / process_ms if result["process_wall_ms"] else None
    )
    result["model_load_ms"] = (manifest or {}).get("model_load_ms")
    result["fallback_load_ms"] = (manifest or {}).get("fallback_load_ms")
    return result


def confound_checks(
    manifests: dict[str, dict[str, Any]],
    cohort_path: Path,
) -> dict[str, Any]:
    off, on = manifests["off"], manifests["on"]
    if bool(off.get("invoice_authorization", False)) != bool(
        on.get("invoice_authorization", False)
    ):
        raise ValueError("Paired arms differ in invoice authorization mode")
    if off.get("experiment_variant", "history-observation-v1") != on.get(
        "experiment_variant", "history-observation-v1"
    ):
        raise ValueError("Paired arms belong to different experimental variants")
    matched, unrecorded = [], []
    for field in SHARED_FIELDS:
        # Legacy history-only manifests predate the opt-in authorization experiment.
        if (
            field in {"invoice_authorization", "experiment_variant"}
            and field not in off
            and field not in on
        ):
            continue
        if field not in off or field not in on:
            unrecorded.append(field)
        elif off[field] != on[field]:
            raise ValueError(f"Paired arms differ in shared setting {field}")
        else:
            matched.append(field)
    history_values = {
        "off": {False, "off", "disabled", "hidden"},
        "on": {True, "on", "enabled", "observed"},
    }
    for arm, manifest in manifests.items():
        if "history_mode" not in manifest:
            unrecorded.append(arm + ":history_mode")
        elif manifest["history_mode"] not in history_values[arm]:
            raise ValueError(f"{arm} manifest has wrong history_mode")
        if manifest.get("cohort_manifest_sha256") not in (None, digest(cohort_path)):
            raise ValueError("Run references a different cohort manifest")
    return {
        "matched_fields": matched,
        "unrecorded_fields": unrecorded,
        "history_difference": {arm: m.get("history_mode") for arm, m in manifests.items()},
        "execution_order": {
            arm: m.get("run_order", "not recorded") for arm, m in manifests.items()
        },
        "timing_caveat": "Bedrock/service load and Docker cache can vary across sequential runs; "
        "one attempt per task cannot establish stable latency differences.",
    }


def analyze(cohort_path: Path, off_path: Path, on_path: Path) -> dict[str, Any]:
    cohort = json.loads(cohort_path.read_text())
    tasks = cohort["tasks"]
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Cohort requires a nonempty tasks mapping")
    paths = {"off": off_path, "on": on_path}
    manifests = {
        arm: json.loads((path / "manifest.json").read_text()) for arm, path in paths.items()
    }
    rows = {arm: unique_rows(path / "results.jsonl") for arm, path in paths.items()}
    expected = set(tasks)
    for arm in paths:
        declared = manifests[arm]["task_ids"]
        if len(declared) != len(set(declared)) or set(declared) != expected:
            raise ValueError(f"{arm} manifest does not cover the entire frozen cohort exactly once")
        if set(rows[arm]) != expected:
            raise ValueError(f"{arm} results lack complete paired cohort coverage")
    checks = confound_checks(manifests, cohort_path)
    families: dict[str, list[str]] = defaultdict(list)
    paired = []
    for task in sorted(tasks):
        metadata = tasks[task]
        family = metadata.get("prototype_id") or metadata.get("group_id")
        if not family:
            raise ValueError(f"Cohort task lacks family identity: {task}")
        families[family].append(task)
        before, after = rows["off"][task], rows["on"][task]
        pair = {
            "task_id": task,
            "family": family,
            "primary_history_probe": bool(metadata.get("primary_history_probe")),
            "unobserved_refund_history_control": bool(
                metadata.get("unobserved_refund_history_control")
            ),
            "clarification_prose_not_scored": bool(metadata.get("clarification_prose_not_scored")),
            "off_verified": verified(before),
            "on_verified": verified(after),
            "off_completed": completed(before),
            "on_completed": completed(after),
            "off_terminal_state_invariants": terminal_invariants(before),
            "on_terminal_state_invariants": terminal_invariants(after),
            "completed_delta": int(completed(after)) - int(completed(before)),
        }
        for key in (
            *COUNTS,
            "authorization_blocked_attempts",
            "end_to_end_policy_ms",
            "total_wall_including_cleanup_ms",
        ):
            legacy_default = (
                0
                if key == "authorization_blocked_attempts"
                and not manifests["off"].get("invoice_authorization")
                else None
            )
            left, right = before.get(key, legacy_default), after.get(key, legacy_default)
            pair["off_" + key], pair["on_" + key] = left, right
            pair[key + "_delta"] = right - left if finite(left) and finite(right) else None
        paired.append(pair)
    arms = {arm: summarize(list(row_map.values()), manifests[arm]) for arm, row_map in rows.items()}
    family_results = {
        family: {arm: summarize([rows[arm][task] for task in identifiers]) for arm in paths}
        for family, identifiers in sorted(families.items())
    }
    slices = {}
    for slice_name in (
        "primary_history_probe",
        "unobserved_refund_history_control",
        "clarification_prose_not_scored",
    ):
        subset = [task for task in tasks if tasks[task].get(slice_name) is True]
        if subset:
            slices[slice_name] = {
                "task_count": len(subset),
                "arms": {arm: summarize([rows[arm][task] for task in subset]) for arm in paths},
            }
    differences: dict[str, Any] = {
        "definition": "ON minus OFF; negative call, error and latency deltas are reductions",
        "improved_tasks": sum(p["completed_delta"] == 1 for p in paired),
        "regressed_tasks": sum(p["completed_delta"] == -1 for p in paired),
        "both_completed": sum(p["off_completed"] and p["on_completed"] for p in paired),
        "neither_completed": sum(not p["off_completed"] and not p["on_completed"] for p in paired),
        "completed_rate_delta": arms["on"]["completed_rate"] - arms["off"]["completed_rate"],
    }
    for key in (
        *COUNTS,
        "authorization_blocked_attempts",
        "end_to_end_policy_ms",
        "total_wall_including_cleanup_ms",
    ):
        values = [pair[key + "_delta"] for pair in paired if finite(pair[key + "_delta"])]
        differences[key] = {
            "paired_tasks": len(values),
            "sum_delta": sum(values) if len(values) == len(tasks) else None,
            "mean_delta": statistics.mean(values) if values else None,
            "median_delta": statistics.median(values) if values else None,
        }
    return {
        "schema_version": "billing-invoice-authorization-history-paired-report-v1"
        if manifests["off"].get("invoice_authorization")
        else "billing-history-paired-report-v1",
        "invoice_authorization": manifests["off"].get("invoice_authorization", False),
        "experiment_variant": manifests["off"].get("experiment_variant", "history-observation-v1"),
        "cohort_id": cohort.get("cohort_id"),
        "task_count": len(tasks),
        "prototype_count": len(families),
        "attempts_per_task_per_arm": 1,
        "confound_checks": checks,
        "arms": arms,
        "paired_differences": differences,
        "per_family": family_results,
        "diagnostic_slices": slices,
        "expected_operation_diagnostic": {
            arm: operation_diagnostic(row_map, tasks) for arm, row_map in rows.items()
        },
        "cohort_limitations": cohort.get("limitations", []),
        "per_task": paired,
        "latency_scopes": LATENCY_SCOPES,
        "limitations": [
            "Author-constructed scenario families are an exploratory targeted diagnostic, not an "
            "independent blind benchmark or a population sample.",
            "Related task variants are not independent evidence; report per-family outcomes.",
            "No training, GPU inference or comparison with the initial fine-tuning run is implied.",
            "Both paired arms use the same invoice-specific authorization parser and execution "
            "guard; history visibility is the paired difference. This follow-up reuses the "
            "previous cohort after inspecting its failures, so it is a targeted regression "
            "diagnostic rather than new holdout evidence."
            if manifests["off"].get("invoice_authorization")
            else "The feature combines visible escalation history with a minimal history-aware "
            "duplicate guard. It does not change the original intent parser.",
            "The refund-history control slice also contains a separate pre-escalated dispute. "
            "Whole-task success may improve by avoiding that duplicate escalation; it does not "
            "demonstrate observation or resolution of the still-hidden refund history. The "
            "requested read-only handling already avoids refunding the drifted invoice.",
            "Deferral, graph termination and verifier pass are distinct; success requires explicit "
            "policy completion plus native verifier pass.",
            "Terminal-state-invariant passes exclude aggregate reward and trajectory. They are "
            "a separate diagnostic: rejected duplicate writes can fail trajectory checks even "
            "when the final database already satisfies every state invariant.",
            "Missing usage/timing measurements remain null. Token totals are provider-reported; "
            "no dollar cost is inferred.",
            "Runtime forbidden-write metrics cover observed-policy violations and disallowed "
            "targets, but do not fully enforce expected operation type on an allowed target. "
            "The separate expected-operation diagnostic uses evaluator-only cohort metadata.",
            "Authorization blocks are refusals before the billing tool executes, not successful "
            "writes or automatic task successes. Blocked tasks are separately classified by "
            "whether they subsequently complete and pass the native verifier. Runtime forbidden "
            "attempt counts can include these blocked requests; forwarded attempts are reported "
            "separately by the posthoc operation diagnostic.",
            "Small-sample p95 and single-run latency changes are descriptive estimates.",
        ],
        "provenance": {
            "cohort_manifest_sha256": digest(cohort_path),
            "reporter_sha256": digest(Path(__file__)),
            "runs": {
                arm: {
                    "manifest_sha256": digest(path / "manifest.json"),
                    "results_sha256": digest(path / "results.jsonl"),
                }
                for arm, path in paths.items()
            },
        },
    }


def markdown(report: dict[str, Any]) -> str:
    authorized = report.get("invoice_authorization", False)
    safety = {arm: values["successful_forbidden_writes"] for arm, values in report["arms"].items()}
    safety_note = (
        f"Successful forbidden writes remain: OFF {safety['off']}, ON {safety['on']}. "
        + (
            "The experimental authorization guard still permits unwanted writes; completion "
            "gains must be read alongside these failures."
            if authorized
            else "The history feature does not repair the unchanged invoice-intent parser; "
            "completion gains must be read alongside these actual unwanted writes."
        )
        if any(value and value > 0 for value in safety.values())
        else "Runtime forbidden-write counts and the separate expected-operation diagnostic "
        "below have different scopes; neither substitutes for native task correctness."
    )
    if authorized and all(value == 0 for value in safety.values()):
        missing = report["expected_operation_diagnostic"]
        safety_note = (
            "No successful forbidden writes were observed. Required operations without a matching "
            "successful tool result remained: "
            f"OFF {missing['off']['missing_successful_expected_operations']}, "
            f"ON {missing['on']['missing_successful_expected_operations']}. "
            "Reduced model use must be read alongside these omissions."
        )
    lines = [
        "# Experimental invoice-authorization history comparison"
        if authorized
        else "# Experimental escalation-history comparison",
        "",
        f"{report['task_count']} paired tasks across {report['prototype_count']} "
        + ("previously evaluated families; " if authorized else "newly authored families; ")
        + "one attempt per task and history mode. Both arms use the same rule+LLM hybrid"
        + (" with invoice-specific authorization." if authorized else "."),
        "",
        safety_note,
        "",
        "| History | Completed + verified | Verifier pass | State invariants only | LLM calls | "
        "Blocked authorization attempts | Forbidden attempts | Forbidden writes | Policy p50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, value in report["arms"].items():
        p50 = value["policy_latency"]["p50_ms"]
        shown = f"{p50:.1f} ms" if p50 is not None else "unavailable"
        lines.append(
            f"| {arm.upper()} | {value['completed_and_verified']}/{value['tasks']} | "
            f"{value['verifier_passes']}/{value['tasks']} | "
            f"{value['terminal_state_invariants_passes']}/"
            f"{value['terminal_state_invariants_observed_tasks']} | {value['llm_calls']} | "
            f"{value['authorization_blocked_attempts']} | "
            f"{value['forbidden_write_attempts']} | {value['successful_forbidden_writes']} | "
            f"{shown} |"
        )
    diff = report["paired_differences"]
    lines.extend(
        [
            "",
            f"Paired completion outcomes: {diff['improved_tasks']} improved, "
            f"{diff['regressed_tasks']} regressed, {diff['both_completed']} completed in both, "
            f"{diff['neither_completed']} completed in neither.",
            "",
            "| Family | OFF completed | ON completed | OFF calls | ON calls |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family, arms in report["per_family"].items():
        off, on = arms["off"], arms["on"]
        lines.append(
            f"| {family} | {off['completed_and_verified']}/{off['tasks']} | "
            f"{on['completed_and_verified']}/{on['tasks']} | "
            f"{off['llm_calls']} | {on['llm_calls']} |"
        )
    lines.extend(
        [
            "",
            "Policy latency includes decision, tools and fallback, while task and process "
            "totals in metrics.json retain their separate scopes. Runtime configuration and "
            "task coverage were checked before comparison. Missing confound metadata: "
            + ", ".join(report["confound_checks"]["unrecorded_fields"] or ["none"])
            + ".",
            "",
        ]
    )
    for name, slice_result in report["diagnostic_slices"].items():
        off, on = slice_result["arms"]["off"], slice_result["arms"]["on"]
        lines.append(
            f"- {name}: {slice_result['task_count']} tasks; completed + verified "
            f"{off['completed_and_verified']} OFF versus {on['completed_and_verified']} ON."
        )
    lines.append("")
    for arm, diagnostic in report["expected_operation_diagnostic"].items():
        lines.append(
            f"- {arm.upper()} posthoc unexpected-operation attempts: "
            f"{diagnostic['unexpected_operation_attempts']}; successful unexpected operations: "
            f"{diagnostic['successful_unexpected_operations']}; expected operations without a "
            "matching successful tool result: "
            f"{diagnostic['missing_successful_expected_operations']} "
            f"across {diagnostic['tasks_missing_successful_expected_operations']} tasks."
        )
    if authorized:
        lines.append("")
        for arm, values in report["arms"].items():
            lines.append(
                f"- {arm.upper()} tasks with tool-execution authorization blocks: "
                f"{values['tasks_with_authorization_blocks']}; subsequently completed + verified: "
                f"{values['completed_after_authorization_block']}; unresolved or failed: "
                f"{values['unresolved_after_authorization_block']}."
            )
    lines.append("")
    lines.extend("- " + limitation for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--off", required=True, type=Path)
    parser.add_argument("--on", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.cohort, args.off, args.on)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "results.md").write_text(markdown(report))
    freeze_path = args.off.parent / "policy-freeze.json"
    if freeze_path.exists():
        (args.output / "policy-freeze.json").write_bytes(freeze_path.read_bytes())
    with (args.output / "paired-tasks.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["per_task"][0]))
        writer.writeheader()
        writer.writerows(report["per_task"])
    print(
        json.dumps(
            {
                "tasks": report["task_count"],
                "families": report["prototype_count"],
                "improved": report["paired_differences"]["improved_tasks"],
                "regressed": report["paired_differences"]["regressed_tasks"],
            }
        )
    )


if __name__ == "__main__":
    main()
