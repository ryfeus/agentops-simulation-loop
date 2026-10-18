"""Report retained paired runs, failures and recoveries without additional inference."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.decision_model.next_evaluate import digest, task_outcomes, write
from experiments.decision_model.next_report import arm_metrics as original_arm_metrics

_spec = importlib.util.spec_from_file_location(
    "v4_offline_audit", Path(__file__).with_name("audit.py")
)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
ARMS = _audit.ARMS
read, records, identity = _audit.read, _audit.records, _audit.identity
ACTION_TOOL = {"refund": "refund_invoice", "escalate": "escalate_dispute"}


def arm_metrics(rows):
    result = original_arm_metrics(rows)
    tasks = {r["task_id"] for r in rows}
    result["tasks_completed_both_attempts"] = sum(
        {r["attempt"] for r in rows if r["task_id"] == task} == {0, 1}
        and all(r["completed_and_verified"] for r in rows if r["task_id"] == task)
        for task in tasks
    )
    return result


def diagnose(rows: list[dict[str, Any]], tasks: dict[str, Any]) -> dict[str, Any]:
    """Distinguish evidence-backed proposal errors from unresolved runtime causes.

    Expected writes are evaluation labels only; this function never runs a policy.
    Guard rejection is retained as an event and is not itself called a model error.
    """
    events, residuals = [], []
    for row in rows:
        categories = set()
        for index, decision in enumerate(row.get("decisions", [])):
            state = decision.get("state", {})
            proposal = decision.get("proposal", {})
            target = decision.get("target_invoice", state.get("target_invoice"))
            expected = tasks[row["task_id"]]["expected_operations"].get(target)
            action = proposal.get("action")
            observation = state.get("observation", {})
            history = observation.get("escalation_history", {})
            invoice = observation.get("invoice", {})
            observed_review = (
                history.get("complete") is True
                and history.get("invoice_id") == target
                and invoice.get("id") == target
                and bool(history.get("records"))
                and all(
                    r.get("invoice_id") == target and r.get("id")
                    for r in history.get("records", [])
                )
            )
            evidence = []
            if row["arm"] == "released":
                if proposal.get("error"):
                    evidence.append("model_inference_error")
                if action in ACTION_TOOL and expected and ACTION_TOOL[action] != expected:
                    evidence.append("model_wrong_operation_vs_frozen_expected_write")
                if (
                    action == "refund"
                    and invoice.get("id") == target
                    and invoice.get("status") != "paid"
                ):
                    evidence.append("model_refund_on_observed_nonpaid_invoice")
                if action == "escalate" and observed_review:
                    evidence.append(
                        "model_duplicate_review_proposal_with_complete_observed_history"
                    )
            resolution = decision.get("invoice_resolution", {})
            resolved = (
                resolution.get("status") == "resolved"
                or decision.get("action_result", {}).get("ok") is True
            )
            unresolved = (
                target in row.get("unresolved_invoices", [])
                or resolution.get("status") == "unresolved"
            )
            auth = state.get("authorization", {})
            expected_action = next((a for a, tool in ACTION_TOOL.items() if tool == expected), None)
            prior_success = any(
                p.get("name") == expected
                and p.get("arguments", {}).get("invoice_id") == target
                and p.get("result", {}).get("ok") is True
                for p in state.get("prior_actions", [])
            )
            if (
                unresolved
                and expected_action
                and not prior_success
                and expected_action not in auth.get("allowed_actions", [])
            ):
                evidence.append(
                    "authorization_or_condition_resolution_candidate_requires_trace_review"
                )
            if unresolved and not expected and observed_review:
                evidence.append("existing_review_completion_candidate_requires_request_review")
            fallback = decision.get("fallback_result", {})
            if isinstance(fallback, dict) and (
                fallback.get("error") or fallback.get("graph_tool_errors")
            ):
                evidence.append("fallback_execution_error")
            if decision.get("fallback_reason") == "authorization_blocked":
                evidence.append("guard_rejected_proposal_routed_to_fallback")
            if evidence or decision.get("prohibited_proposal") or unresolved:
                categories.update(evidence)
                events.append(
                    {
                        "trial": list(identity(row)),
                        "decision_index": index,
                        "target_invoice": target,
                        "expected_operation": expected,
                        "evidence_categories": evidence,
                        "resolved": resolved,
                        "trial_completed_and_verified": row.get("completed_and_verified"),
                        "raw_decision": decision,
                    }
                )
        tool_failures = [
            o for o in row.get("tool_observations", []) if o.get("result", {}).get("ok") is False
        ]
        if tool_failures:
            categories.add("tool_or_guard_failure")
        if row.get("error") or row.get("verifier_error"):
            categories.add("execution_or_verification_infrastructure_error")
        if not row.get("completed_and_verified"):
            residuals.append(
                {
                    "trial": list(identity(row)),
                    "categories": sorted(categories) or ["unattributed_requires_trace_review"],
                    "unresolved_invoices": row.get("unresolved_invoices", []),
                    "missing_required_operation_details": row.get(
                        "missing_required_operation_details", []
                    ),
                    "fallback_error": row.get("fallback_error"),
                    "error": row.get("error"),
                    "verifier_error": row.get("verifier_error"),
                    "tool_failures": tool_failures,
                    "native_verifier": row.get("verifier"),
                }
            )
    return {
        "residual_trials": residuals,
        "proposal_and_recovery_events": events,
        "event_category_counts": dict(Counter(c for e in events for c in e["evidence_categories"])),
        "guard_rejected_recovered_decisions": sum(
            "guard_rejected_proposal_routed_to_fallback" in e["evidence_categories"]
            and e["resolved"]
            for e in events
        ),
        "notes": [
            "Categories can overlap and are not an exhaustive causal attribution.",
            "Guard rejection alone is not a model error: parsing "
            "or unresolved conditions may be responsible.",
            "Candidate parser/completion causes require the retained "
            "request and observation to be reviewed.",
            "Recovered wrong proposals remain counted even when the native task passes.",
            "No policy, parser, model or tool is executed by this offline diagnosis.",
        ],
    }


def analyze(run: Path) -> dict[str, Any]:
    manifest = read(run / "manifest.json")
    cohort = read(run / "cohort-manifest.json")
    tasks = cohort["tasks"]
    rows = records(run / "results.jsonl")
    if not isinstance(tasks, dict):
        tasks = {t["task_id"]: t for t in tasks}
    # Recompute instead of trusting aggregate outcome fields, retaining failures.
    measured = [{**row, **task_outcomes(row, tasks[row["task_id"]])} for row in rows]
    indexed = {identity(row): row for row in measured}
    families = defaultdict(list)
    for task in manifest["task_ids"]:
        families[tasks[task]["composition_root"]].append(task)
    pairs = []
    for task in manifest["task_ids"]:
        for attempt in range(manifest["repeats"]):
            left, right = (
                indexed.get((task, "released", attempt)),
                indexed.get((task, "llm", attempt)),
            )
            if left is None or right is None:
                continue
            pairs.append(
                {
                    "task_id": task,
                    "attempt": attempt,
                    "composition_root": tasks[task]["composition_root"],
                    "released_completed": left["completed_and_verified"],
                    "llm_completed": right["completed_and_verified"],
                    "completion_delta": int(left["completed_and_verified"])
                    - int(right["completed_and_verified"]),
                }
            )
    groups = {arm: [r for r in measured if r["arm"] == arm] for arm in ARMS}
    return {
        "run": str(run),
        "cohort_previously_exposed": manifest["cohort_previously_exposed"],
        "complete": manifest.get("complete"),
        "terminal_error": manifest.get("error"),
        "total_trials": len(rows),
        "planned_trials": manifest.get("planned_trials"),
        "distinct_tasks": len(manifest["task_ids"]),
        "distinct_families": len(families),
        "arms": {arm: arm_metrics(group) if group else {} for arm, group in groups.items()},
        "per_family": {
            family: {
                arm: arm_metrics(subset)
                if (subset := [r for r in group if r["task_id"] in identifiers])
                else {}
                for arm, group in groups.items()
            }
            for family, identifiers in families.items()
        },
        "pairs": pairs,
        "paired_comparison": {
            "released_wins": sum(p["completion_delta"] == 1 for p in pairs),
            "released_losses": sum(p["completion_delta"] == -1 for p in pairs),
            "ties": sum(p["completion_delta"] == 0 for p in pairs),
        },
        "diagnosis": diagnose(measured, tasks),
        "resource_accounting": {
            "execution_wall_seconds": manifest.get("run_total_wall_ms", 0) / 1000,
            "max_job_seconds": manifest.get("max_job_seconds"),
            "model_load_ms": manifest.get("model_load_ms"),
            "fallback_load_ms": manifest.get("fallback_load_ms"),
            "owned_environments_deleted": sum(
                r.get("cleanup") == "deleted-owned-environment" for r in rows
            ),
            "usage": {
                field: {
                    "known_total": sum(r[field] for r in rows if isinstance(r.get(field), int)),
                    "missing_trials": sum(not isinstance(r.get(field), int) for r in rows),
                }
                for field in ("llm_calls", "llm_input_tokens", "llm_output_tokens")
            },
            "training_steps": manifest.get("training_steps"),
            "calibration_fits": manifest.get("calibration_fits"),
            "device": manifest.get("device"),
            "new_training_workers": 0,
            "notes": [
                "Returned provider usage may omit failed requests; no dollar total is inferred.",
                "Policy latency excludes setup, verification and cleanup; "
                "execution wall time includes startup.",
                "Deletion counts are retained runner claims; independent "
                "inventories establish preexisting preservation.",
            ],
        },
        "provenance": {
            name: digest(run / name) if (run / name).is_file() else None
            for name in ("manifest.json", "results.jsonl", "freeze.json", "cohort-manifest.json")
        },
    }


def historical_comparison(current: Path, historical: Path) -> dict[str, Any]:
    before, after = records(historical / "results.jsonl"), records(current / "results.jsonl")
    old = {identity(r): r for r in before}
    new = {identity(r): r for r in after}
    common = sorted(set(old) & set(new))
    old_tasks = read(historical / "cohort-manifest.json")["tasks"]
    new_tasks = read(current / "cohort-manifest.json")["tasks"]
    identity_mismatches = [
        task
        for task in {k[0] for k in common}
        if any(
            old_tasks[task].get(f) != new_tasks[task].get(f)
            for f in ("execution_task_sha256", "scenario_sha256", "expected_operations")
        )
    ]
    result = {
        "historical_run": str(historical),
        "matched_trials": len(common),
        "task_identity_mismatches": identity_mismatches,
        "arms": {},
        "caveat": "Previously exposed engineering comparison; "
        "earlier outcomes informed repairs. Historical runs "
        "are not contemporaneous randomized controls; provider "
        "variation and runtime differences confound causal estimates. "
        "Only identical task/arm/attempt identities are matched; "
        "unmatched new repetitions are excluded.",
    }
    for arm in ARMS:
        keys = [key for key in common if key[1] == arm]
        result["arms"][arm] = {
            "matched_trials": len(keys),
            "before_completed": sum(old[k].get("completed_and_verified") is True for k in keys),
            "after_completed": sum(new[k].get("completed_and_verified") is True for k in keys),
            "wins": sum(
                not old[k].get("completed_and_verified") and new[k].get("completed_and_verified")
                for k in keys
            ),
            "losses": sum(
                old[k].get("completed_and_verified") and not new[k].get("completed_and_verified")
                for k in keys
            ),
        }
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Released decider versus LLM: runtime repair evaluation",
        "",
        "Weights and calibration were unchanged. These results "
        "measure repaired execution and routing, not learned "
        "improvement.",
        "",
    ]
    for label in ("regression", "fresh"):
        result = report.get(label)
        if not result:
            lines += [f"Fresh held-out evaluation: {report['fresh_status']}.", ""]
            continue
        lines += [
            f"## {label.title()}",
            "",
            f"{result['total_trials']}/{result['planned_trials']} "
            f"retained trials across {result['distinct_tasks']} tasks "
            f"and {result['distinct_families']} families. Complete: {result['complete']}. "
            f"Terminal error: {result['terminal_error']}. Previously "
            f"exposed: {result['cohort_previously_exposed']}.",
            "",
            "| Arm | Strict completion | Native passes | Required "
            "writes | Unwanted writes | LLM calls | Input/output "
            "tokens | Policy p50/p95 ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for arm, m in result["arms"].items():
            if not m:
                continue
            latency = m["end_to_end_policy_ms"]
            lines.append(
                f"| {arm} | {m['completed_and_verified']}/{m['trials']} "
                f"| {m['verifier_passes']}/{m['trials']} | "
                f"{m['successful_required_operations']}/{m['required_operations']} "
                f"| {m['successful_unwanted_writes']} | {m['llm_calls']} "
                f"| {m['llm_input_tokens']}/{m['llm_output_tokens']} "
                f"| {latency['p50']}/{latency['p95']} |"
            )
        pair = result["paired_comparison"]
        lines += [
            "",
            f"Matched released-vs-LLM pairs: {pair['released_wins']} "
            f"wins, {pair['released_losses']} losses, {pair['ties']} "
            f"ties.",
            "",
            "| Family | Released completion | LLM completion |",
            "|---|---:|---:|",
        ]
        for family, groups in result["per_family"].items():
            lines.append(
                "| "
                + family
                + " | "
                + " | ".join(
                    f"{groups[a].get('completed_and_verified', 0)}/{groups[a].get('trials', 0)}"
                    for a in ARMS
                )
                + " |"
            )
        diagnosis = result["diagnosis"]
        lines += [
            "",
            f"Residual failed trials: {len(diagnosis['residual_trials'])}. "
            "Guard-rejected decisions recovered: "
            f"{diagnosis['guard_rejected_recovered_decisions']}.",
            "Raw proposals and observation-backed attribution are "
            "retained in metrics.json, including recovered errors. "
            "Parser/completion candidates are explicitly distinguished "
            "from confirmed proposal contradictions.",
            "",
        ]
    lines += ["## Historical matched subsets", ""]
    for item in report.get("historical_comparisons", []):
        lines.append(
            f"- {item['historical_run']}: {item['matched_trials']} matched trials. "
            + "; ".join(
                f"{a} {m['before_completed']}→{m['after_completed']}/{m['matched_trials']} "
                f"({m['wins']} wins, {m['losses']} losses)"
                for a, m in item["arms"].items()
            )
        )
    lines += [
        "",
        "Historical results are exposed engineering comparisons, "
        "with provider variation and runtime differences; they "
        "do not establish fresh generalization or isolated causal "
        "effects.",
        "",
        "Two attempts and variants within a family are correlated. "
        "Fresh authored families share known billing primitives "
        "and are not an external benchmark. Native verification "
        "does not fully grade reporting prose. Zero observed "
        "unwanted writes applies only to retained determinate "
        "evidence.",
        "",
        "No additional training worker was used. Provider tokens "
        "are returned usage, not an AWS bill; failed calls may "
        "omit usage. Detailed latency, calls, tokens, cleanup "
        "claims and integrity evidence are retained in JSON.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(".artifacts/decision-model-generalization-v4")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/decision-model-generalization-v4")
    )
    args = parser.parse_args()
    regression_path = args.root / "regression-paid"
    regression_audit = _audit.audit_run(
        regression_path, expected_tasks=36, exposed=True, expected_repeats=1
    )
    result = {
        "regression": analyze(regression_path),
        "regression_audit": regression_audit,
        "fresh": None,
        "fresh_status": "blocked by strict regression acceptance gate",
        "historical_comparisons": [],
    }
    fresh = args.root / "final-harbor"
    if (fresh / "manifest.json").exists():
        if not regression_audit["strict_gate_passed"]:
            raise ValueError("Fresh inference exists despite failed strict regression gate")
        result["fresh"] = analyze(fresh)
        result["fresh_audit"] = _audit.audit_run(fresh, expected_tasks=12, exposed=False)
        result["fresh_status"] = "executed; inspect audit and measured outcomes"
    elif regression_audit["strict_gate_passed"]:
        result["fresh_status"] = "not yet executed"
    for name in ("regression-paid", "final-harbor"):
        historical = Path(".artifacts/decision-model-generalization-v3") / name
        if (historical / "results.jsonl").exists():
            result["historical_comparisons"].append(
                {
                    **historical_comparison(regression_path, historical),
                    "historical_audit": _audit.audit_historical(historical),
                }
            )
    before, after = args.output / "docker-before.jsonl", args.output / "docker-after.jsonl"
    paid_seconds = sum(
        result[label]["resource_accounting"]["execution_wall_seconds"]
        for label in ("regression", "fresh")
        if result.get(label)
    )
    result["combined_paid_execution"] = {
        "seconds": paid_seconds,
        "limit_seconds": 3600,
        "within_limit": 0 <= paid_seconds <= 3600,
    }
    result["container_inventory_audit"] = (
        _audit.audit_container_inventories(before, after)
        if before.exists() and after.exists()
        else {"status": "final inventory unavailable"}
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "metrics.json", result)
    write(args.output / "regression-audit.json", regression_audit)
    if "fresh_audit" in result:
        write(args.output / "audit.json", result["fresh_audit"])
    (args.output / "results.md").write_text(markdown(result))
    print(
        json.dumps(
            {
                "regression_integrity": regression_audit["passed"],
                "regression_gate": regression_audit["strict_gate_passed"],
                "fresh_status": result["fresh_status"],
            }
        )
    )


if __name__ == "__main__":
    main()
