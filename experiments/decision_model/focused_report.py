"""Offline integrity and strict acceptance audit; never invokes a policy or service."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.decision_model.focused_evaluate import ARMS, planned_order
from experiments.decision_model.next_evaluate import (
    digest,
    summarize,
    task_outcomes,
    validate_task,
    write,
)


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def identity(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["task_id"], row["arm"], row["attempt"]


def strict_row(row: dict[str, Any]) -> bool:
    argument_errors = any(
        item.get("result", {}).get("error_type") == "InvoiceToolArgumentsError"
        for item in row.get("tool_observations", [])
    )
    graph_errors = row.get("graph_tool_errors") or any(
        d.get("fallback_result", {}).get("graph_tool_errors")
        for d in row.get("decisions", [])
        if isinstance(d.get("fallback_result"), dict)
    )
    return bool(
        row.get("completed_and_verified") is True
        and row.get("verifier_pass") is True
        and row.get("successful_required_operations") == row.get("required_operations")
        and row.get("missing_required_operations") == 0
        and row.get("successful_unwanted_writes") == 0
        and row.get("tool_evidence_complete") is True
        and not row.get("indeterminate_tool_calls")
        and not row.get("error")
        and not row.get("verifier_error")
        and not row.get("invoice_scope_blocked_attempts")
        and not argument_errors
        and not graph_errors
    )


def audit_run(
    run: Path, *, expected_tasks: int, exposed: bool, expected_repeats: int = 2
) -> dict[str, Any]:
    failures: list[str] = []
    hashes: dict[str, str] = {}

    def check(ok: bool, description: str) -> None:
        if not ok:
            failures.append(description)

    def hashed(path: Path, expected: str | None, description: str) -> None:
        if not path.is_file():
            failures.append(description + ": missing " + str(path))
            return
        hashes[str(path)] = digest(path)
        check(hashes[str(path)] == expected, description + ": " + str(path))

    manifest = read(run / "manifest.json")
    freeze = read(run / "freeze.json")
    cohort = read(run / "cohort-manifest.json")
    rows = records(run / "results.jsonl")
    tasks = cohort["tasks"]
    if not isinstance(tasks, dict):
        tasks = {t["task_id"]: t for t in tasks}
    task_ids = freeze["task_ids"]
    order = planned_order(task_ids)
    expected = set(order)
    indexed = {identity(row): row for row in rows}
    check(manifest.get("complete") is True and not manifest.get("error"), "bounded run completed")
    check(len(task_ids) == len(set(task_ids)) == expected_tasks, "frozen distinct task count")
    check(manifest.get("task_ids") == task_ids, "manifest task selection equals freeze")
    check(len(rows) == len(indexed) == len(expected), "unique complete trial coverage")
    check(set(indexed) == expected, "exact paired identities; no selective retries")
    check(
        manifest.get("planned_trials") == freeze.get("planned_trials") == len(expected),
        "planned trial count",
    )
    check(
        manifest.get("repeats") == freeze.get("repeats") == expected_repeats,
        "attempt count predeclared",
    )
    check([identity(r) for r in rows] == order, "aggregate follows alternating interleaving")
    executed = manifest.get("execution_order", [])
    check([identity(r) for r in executed] == order, "execution follows frozen interleaving")
    check(
        [r.get("position") for r in executed] == list(range(len(order))),
        "contiguous execution positions",
    )
    check(
        [r.get("execution_position") for r in rows] == list(range(len(order))),
        "aggregate positions",
    )
    check(freeze.get("optimizer_steps", 0) > 0, "positive trained steps")
    check(
        all(
            freeze.get("weight_changes", {}).get("changed_elements", {}).get(k, 0) > 0
            for k in ("adapter", "head")
        ),
        "nonzero adapter and head updates",
    )
    for document in (manifest, freeze, cohort):
        check(
            document.get("cohort_previously_exposed") is exposed, "consistent exposure declaration"
        )
    for field, value in (
        ("device", freeze.get("device")),
        ("max_tool_calls", 16),
        ("max_llm_calls", 16),
        ("fallback_timeout", 120),
        ("task_timeout", 180),
    ):
        check(manifest.get(field) == value, "execution bound " + field)
    check(
        0 < manifest.get("max_job_seconds", 0) <= 3600
        and manifest.get("max_job_seconds") == freeze.get("max_job_seconds"),
        "frozen bounded wall time",
    )
    elapsed = manifest.get("run_total_wall_ms")
    check(
        isinstance(elapsed, (int, float))
        and 0 <= elapsed <= manifest.get("max_job_seconds", 0) * 1000,
        "measured execution within frozen wall-time bound",
    )
    check(manifest.get("thresholds") == freeze.get("thresholds"), "frozen thresholds")
    check(set(freeze.get("thresholds", {})) == set(ARMS), "exact three arms")
    hashed(run / "freeze.json", manifest.get("freeze_sha256"), "manifest binds freeze")
    hashed(
        run / "cohort-manifest.json",
        manifest.get("cohort_manifest_sha256"),
        "manifest binds cohort",
    )
    check(
        freeze.get("cohort_manifest_sha256") == digest(run / "cohort-manifest.json"),
        "freeze binds cohort",
    )
    source_hashes = freeze.get("source_sha256", {})
    check(bool(source_hashes), "nonempty source lock")
    check(
        len({Path(p).name for p in source_hashes}) == len(source_hashes),
        "source snapshot names unambiguous",
    )
    for source, value in source_hashes.items():
        # Deliberately audit retained bytes, not today's potentially changed source tree.
        hashed(run / ("source-" + Path(source).name), value, "retained source snapshot")
    for field in ("artifact_sha256", "cohort_evidence_sha256"):
        for source, value in freeze.get(field, {}).items():
            hashed(Path(source), value, "frozen " + field)
    check(bool(freeze.get("artifact_sha256")), "nonempty model/configuration artifact lock")
    for task_id in task_ids:
        try:
            check(tasks[task_id].get("split") == "test", "test split: " + task_id)
            validate_task(tasks[task_id])
        except Exception as exc:
            failures.append(f"native task identity {task_id}: {exc}")
    if not exposed:
        for field in (
            "pre_freeze_policy_rollouts",
            "pre_freeze_model_inferences",
            "pre_freeze_authorization_parser_checks",
            "pre_freeze_teacher_collections",
        ):
            check(cohort.get(field) == 0, "fresh provenance " + field)
        access_path = run / "evaluation-access.json"
        if access_path.exists():
            access = read(access_path)
            hashed(
                access_path,
                manifest.get("evaluation_access_sha256"),
                "retained first-access marker",
            )
            check(
                access.get("cohort_manifest_sha256") == manifest.get("cohort_manifest_sha256")
                and access.get("freeze_sha256") == manifest.get("freeze_sha256")
                and access.get("output") == str(run.resolve())
                and access.get("claimed_before_model_initialization") is True,
                "access marker binds frozen cohort before model initialization",
            )
            if freeze.get("evaluation_access_path"):
                hashed(
                    Path(freeze["evaluation_access_path"]),
                    manifest.get("evaluation_access_sha256"),
                    "external first-access marker",
                )
        else:
            failures.append("missing prospective first-access marker")
        novelty = cohort.get("novelty_audit", {})
        check(
            bool(novelty)
            and all(
                novelty.get(k) == 0
                for k in (
                    "root_collisions",
                    "structural_collisions",
                    "normalized_instruction_collisions",
                )
            ),
            "fresh family novelty evidence",
        )
    for row in rows:
        key = str(identity(row))
        if row["task_id"] not in tasks:
            continue
        measured = task_outcomes(row, tasks[row["task_id"]])
        check(
            all(row.get(k) == v for k, v in measured.items()),
            "outcomes from tool/native evidence " + key,
        )
        decisions = row.get("decisions", [])
        observations = row.get("tool_observations", [])
        for field, value in (
            ("prohibited_proposals", sum(bool(d.get("prohibited_proposal")) for d in decisions)),
            (
                "authorization_blocked_attempts",
                sum(bool(o.get("authorization_blocked")) for o in observations),
            ),
            (
                "invoice_scope_blocked_attempts",
                sum(bool(o.get("invoice_scope_blocked")) for o in observations),
            ),
        ):
            check(row.get(field, 0) == value, "recomputed " + field + " " + key)
        check(
            row.get("tool_call_count") == len(observations),
            "tool call count from observations " + key,
        )
        check(
            all(
                isinstance(row.get(k), int) and 0 <= row[k] <= 16
                for k in ("tool_call_count", "llm_calls")
            ),
            "episode budgets " + key,
        )
        check(row.get("invoice_local_fallback") is True, "invoice-local execution " + key)
        check(
            row.get("cleanup") == "deleted-owned-environment", "owned-only deletion record " + key
        )
        check(
            not any(
                (o.get("authorization_blocked") or o.get("invoice_scope_blocked"))
                and o.get("result", {}).get("ok") is True
                for o in observations
            ),
            "blocked tool cannot succeed " + key,
        )
        fallback_targets = [d.get("target_invoice") for d in decisions if d.get("fallback_reason")]
        check(
            row.get("fallback_count", 0) == len(fallback_targets),
            "fallback count from decision evidence " + key,
        )
        check(
            all(n == 1 for n in Counter(fallback_targets).values()),
            "one fallback continuation per invoice " + key,
        )
    if rows:
        for arm in ARMS:
            group = [row for row in rows if row["arm"] == arm]
            if group:
                check(
                    manifest.get("arms", {}).get(arm) == summarize(group),
                    "manifest arm counters recompute " + arm,
                )
    native_hashes: dict[str, str] = {}
    trial_keys = []
    for path in sorted((run / "trials").glob("*/results.jsonl")):
        values = records(path)
        check(len(values) == 1, "one result per native trial " + str(path))
        for row in values:
            key = identity(row)
            trial_keys.append(key)
            check(
                {k: v for k, v in indexed.get(key, {}).items() if k != "execution_position"} == row,
                "trial equals aggregate " + str(key),
            )
            native_hashes[str(path)] = digest(path)
            for name, field in (("reward.json", "rewards"), ("diagnostics.json", "diagnostics")):
                native = path.parent / row["task_id"] / "verifier" / name
                if native.exists():
                    check(
                        read(native) == row.get("verifier", {}).get(field),
                        "native evidence agrees " + str(native),
                    )
                    native_hashes[str(native)] = digest(native)
                else:
                    failures.append("missing native evidence: " + str(native))
    check(
        len(trial_keys) == len(set(trial_keys)) == len(expected) and set(trial_keys) == expected,
        "exact native record coverage",
    )
    check(len(native_hashes) == len(expected) * 3, "complete native evidence files")
    quality = [list(identity(row)) for row in rows if not strict_row(row)]
    return {
        "passed": not failures,
        "failures": failures,
        "strict_gate_passed": not failures and not quality,
        "strict_failure_trials": quality,
        "policy_failures_are_integrity_failures": False,
        "retained_trials": len(rows),
        "expected_trials": len(expected),
        "native_evidence_files_verified": len(native_hashes),
        "native_evidence_sha256": native_hashes,
        "frozen_evidence_sha256": hashes,
        "cohort_previously_exposed": exposed,
        "cleanup_evidence_limit": "Per-trial runner deletion "
        "records; independently compare Docker inventories before "
        "claiming preexisting containers preserved.",
    }


def audit_container_inventories(before: Path, after: Path) -> dict[str, Any]:
    """Read only: inventories cannot prove absence of transient unrelated deletions."""
    old = {row["ID"] for row in records(before)}
    new = {row["ID"] for row in records(after)}
    return {
        "passed": old == new,
        "preexisting_ids": sorted(old),
        "postrun_ids": sorted(new),
        "missing_preexisting": sorted(old - new),
        "additional_containers": sorted(new - old),
        "evidence_sha256": {str(p): digest(p) for p in (before, after)},
        "limitation": (
            "Inventories plus per-trial owned-deletion records; no Docker action is taken."
        ),
    }


def compare(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    """Paired denominators only; report incomplete coverage separately."""
    indexed = {(r["task_id"], r["attempt"], r["arm"]): r for r in rows}
    keys = sorted({(r["task_id"], r["attempt"]) for r in rows})
    pairs = [
        (indexed[(t, a, left)], indexed[(t, a, right)])
        for t, a in keys
        if (t, a, left) in indexed and (t, a, right) in indexed
    ]
    wins = sum(
        bool(a["completed_and_verified"]) > bool(b["completed_and_verified"]) for a, b in pairs
    )
    losses = sum(
        bool(a["completed_and_verified"]) < bool(b["completed_and_verified"]) for a, b in pairs
    )
    left_fallbacks = sum(a.get("fallback_count", 0) for a, _ in pairs)
    right_fallbacks = sum(b.get("fallback_count", 0) for _, b in pairs)
    reduction = 1 - left_fallbacks / right_fallbacks if right_fallbacks else None
    return dict(
        left=left,
        right=right,
        pairs=len(pairs),
        wins=wins,
        losses=losses,
        ties=len(pairs) - wins - losses,
        completion_delta=wins - losses,
        fallback_continuations_left=left_fallbacks,
        fallback_continuations_right=right_fallbacks,
        fallback_reduction_fraction=reduction,
        llm_calls_delta=sum(a.get("llm_calls", 0) - b.get("llm_calls", 0) for a, b in pairs),
        target_met=(
            len(pairs) == 24
            and losses == 0
            and reduction is not None
            and reduction >= 0.2
            and all(a.get("successful_unwanted_writes") == 0 for a, _ in pairs)
        ),
        no_individual_completion_losses=losses == 0,
    )


def report(
    run: Path, output: Path, before: Path | None = None, after: Path | None = None
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_run(run, expected_tasks=12, exposed=False)
    rows = records(run / "results.jsonl")
    manifest = read(run / "manifest.json")
    cohort = read(run / "cohort-manifest.json")
    tasks = cohort["tasks"]
    if not isinstance(tasks, dict):
        tasks = {t["task_id"]: t for t in tasks}
    arms = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        if not group:
            arms[arm] = {"tasks": 0}
            continue
        value = summarize(group)
        latencies = [r["total_wall_ms"] for r in group]
        value.update(
            total_wall_ms=sum(latencies),
            median_wall_ms=statistics.median(latencies),
            tool_calls=sum(r.get("tool_call_count", 0) for r in group),
        )
        arms[arm] = value
    pairs = {
        f"{a}_vs_{b}": compare(rows, a, b)
        for a, b in (("tuned", "released"), ("released", "llm"), ("tuned", "llm"))
    }
    family = {}
    for root in sorted({t["composition_root"] for t in tasks.values()}):
        group = [r for r in rows if tasks[r["task_id"]]["composition_root"] == root]
        family[root] = {
            arm: summarize([r for r in group if r["arm"] == arm])
            for arm in ARMS
            if any(r["arm"] == arm for r in group)
        }
    residuals = [
        {
            key: r.get(key)
            for key in (
                "task_id",
                "arm",
                "attempt",
                "execution_position",
                "error",
                "verifier_error",
                "unresolved_invoices",
                "missing_required_operation_details",
                "fallback_error",
                "successful_unwanted_writes",
                "authorization_blocked_attempts",
            )
        }
        for r in rows
        if not strict_row(r)
    ]
    containers = audit_container_inventories(before, after) if before and after else None
    metrics = dict(
        arms=arms,
        paired=pairs,
        families=family,
        residual_failures=residuals,
        integrity_passed=audit["passed"],
        integrity_failures=audit["failures"],
        native_evidence_files_verified=audit["native_evidence_files_verified"],
        planned_trials=72,
        retained_trials=len(rows),
        containers=containers,
        run_total_wall_ms=manifest.get("run_total_wall_ms"),
        target_met=audit["passed"] and pairs["tuned_vs_released"]["target_met"],
    )
    write(output / "audit.json", audit)
    write(output / "metrics.json", metrics)
    lines = [
        "# Focused fine-tuning comparison",
        "",
        f"Retained {len(rows)}/72 trials; native evidence integrity: {audit['passed']}.",
        "",
        "| Arm | Completed | Native pass | Required writes | Unwanted | Fallbacks | LLM calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, v in arms.items():
        lines.append(
            f"| {arm} | {v.get('completed_and_verified', 0)}/24 | "
            f"{v.get('verifier_passes', 0)} | "
            f"{v.get('successful_required_operations', 0)}/{v.get('required_operations', 0)} | "
            f"{v.get('successful_unwanted_writes', 0)} | "
            f"{v.get('fallback_continuations', 0)} | {v.get('llm_calls')} |"
        )
    p = pairs["tuned_vs_released"]
    lines.extend(
        [
            "",
            f"Tuned versus released: {p['wins']} completion wins, {p['losses']} losses "
            f"across {p['pairs']} matched task attempts. Fallback reduction: "
            f"{p['fallback_reduction_fraction']}. Predeclared target met: {metrics['target_met']}.",
            "",
            "Target: at least 20% fewer invoice fallback continuations, no paired task "
            "completion losses, and zero unwanted writes. LLM starts are reported separately.",
            "",
            "This is a focused authored billing demonstration; variants and repeats are "
            "correlated. Paired counts do not establish statistical noninferiority. "
            "Native verification does not grade every prose claim.",
            "",
            "See metrics.json for tokens, latency, family outcomes and residual failures; "
            "audit.json binds every trial to native evidence and frozen artifacts.",
            "",
        ]
    )
    (output / "results.md").write_text("\n".join(lines))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker-before", type=Path)
    parser.add_argument("--docker-after", type=Path)
    args = parser.parse_args()
    result = report(args.run, args.output, args.docker_before, args.docker_after)
    print(json.dumps({k: result[k] for k in ("integrity_passed", "retained_trials", "target_met")}))
    if not result["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
