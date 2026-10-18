"""Audit the completed prospective generalization run; never invokes models, tools, or Docker.

Run from the repository root after the paid runner exits. Integrity failures are
written to audit.json before a nonzero exit. Policy failures remain in accounting
and are never excluded to improve success rates.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.decision_model.generalization_report import ARMS, analyze
from experiments.decision_model.next_evaluate import (
    check_final_freeze,
    digest,
    final_source_paths,
    validate_task,
    write,
)


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def identity(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["task_id"], row["arm"], row["attempt"]


def model_and_execution_artifacts(freeze: dict[str, Any]) -> dict[str, str]:
    """Exclude only explicitly named reference documents, not model/configuration files."""
    references = {freeze.get("baseline_reference_freeze_path"), freeze.get("runtime_lock_path")}
    return {
        path: value for path, value in freeze["artifact_sha256"].items() if path not in references
    }


def audit_regression(root: Path, bound_evidence: dict[str, str]) -> dict[str, Any]:
    """Read retained exposed-case evidence only; never execute or retry a policy."""
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    for path, expected_hash in bound_evidence.items():
        check(digest(Path(path)) == expected_hash, "locked regression evidence hash: " + path)
    paid = root / "regression-paid"
    manifest_path, result_path = paid / "manifest.json", paid / "results.jsonl"
    for path in (manifest_path, result_path):
        check(
            bound_evidence.get(str(path.resolve())) == digest(path),
            "runtime lock must bind exposed paid regression " + path.name,
        )
    manifest = read(manifest_path)
    rows = records(result_path)
    cohort_path = paid / "cohort-manifest.json"
    cohort = read(cohort_path)
    check_final_freeze(read(paid / "freeze.json"), cohort_path)
    tasks = cohort["tasks"]
    expected = {(task_id, arm, 0) for task_id in tasks for arm in ARMS}
    check(
        manifest.get("complete") is True and manifest.get("error") is None,
        "exposed paid regression must complete without terminal error",
    )
    check(
        manifest.get("cohort_previously_exposed") is True
        and cohort.get("cohort_id") == "billing-generalization-v1",
        "paid regression must identify previously exposed generalization-v1 cases",
    )
    check(
        manifest.get("planned_trials") == len(rows) == len(expected) == 24
        and {identity(row) for row in rows} == expected,
        "all24 exposed paid pairs must be retained exactly once",
    )
    check(
        manifest.get("max_job_seconds") == 900
        and manifest.get("training_steps") == manifest.get("calibration_fits") == 0,
        "exposed regression has900second budget with no fitting",
    )
    check(
        len(manifest.get("execution_order", [])) == len(rows)
        and [identity(row) for row in rows]
        == [identity(entry) for entry in manifest.get("execution_order", [])],
        "paid regression aggregate and execution order agree",
    )
    from experiments.decision_model.next_evaluate import task_outcomes

    indexed = {identity(row): row for row in rows}
    for row in rows:
        measured = task_outcomes(row, tasks[row["task_id"]])
        check(
            all(row.get(key) == value for key, value in measured.items()),
            "paid regression outcomes independently match evidence: " + str(identity(row)),
        )
    argument_violation_trials = [
        identity(row)
        for row in rows
        if any(
            observation.get("result", {}).get("error_type") == "InvoiceToolArgumentsError"
            for observation in row.get("tool_observations", [])
        )
        or row.get("graph_tool_errors")
        or any(
            decision.get("fallback_result", {}).get("graph_tool_errors")
            for decision in row.get("decisions", [])
            if isinstance(decision.get("fallback_result"), dict)
        )
    ]
    check(
        not argument_violation_trials, "zero exposed regression tool-argument or graph-tool errors"
    )
    core_gate_passed = not argument_violation_trials and all(
        row.get("verifier_pass") is True
        and row.get("missing_required_operations") == 0
        and row.get("successful_required_operations") == row.get("required_operations")
        and row.get("successful_unwanted_writes") == 0
        and row.get("invoice_scope_blocked_attempts", 0) == 0
        and row.get("tool_evidence_complete") is True
        and not row.get("indeterminate_tool_calls")
        and not row.get("error")
        and not row.get("verifier_error")
        for row in rows
    )
    check(
        core_gate_passed,
        "exposed regression core gate: native pass, required writes, zero unwanted/scope "
        "violations and complete determinate tool evidence",
    )
    strict_quality_passed = all(row.get("completed_and_verified") is True for row in rows)
    check(
        all(row.get("cleanup") == "deleted-owned-environment" for row in rows),
        "all exposed regression environments have deletion evidence",
    )
    check(
        all(
            isinstance(row.get("tool_call_count"), int)
            and 0 <= row["tool_call_count"] <= 16
            and isinstance(row.get("llm_calls"), int)
            and 0 <= row["llm_calls"] <= 16
            for row in rows
        ),
        "exposed paid tool and LLM bounds respected",
    )
    hashes, trial_identities = {}, []
    for path in sorted((paid / "trials").glob("*/results.jsonl")):
        values = records(path)
        check(len(values) == 1, "one retained regression record per native trial")
        for row in values:
            key = identity(row)
            trial_identities.append(key)
            check(
                {k: v for k, v in indexed.get(key, {}).items() if k != "execution_position"} == row,
                "regression aggregate matches native trial " + str(key),
            )
            verifier = path.parent / row["task_id"] / "verifier"
            for name, field in (("reward.json", "rewards"), ("diagnostics.json", "diagnostics")):
                native = verifier / name
                check(
                    read(native) == row.get("verifier", {}).get(field),
                    "regression native verifier evidence matches " + str(key) + "/" + name,
                )
                hashes[str(native)] = digest(native)
            hashes[str(path)] = digest(path)
    check(
        len(trial_identities) == len(set(trial_identities)) == 24 and len(hashes) == 72,
        "72 exposed regression native record/reward/diagnostic files verified",
    )
    preflights = [Path(path) for path in bound_evidence if Path(path).name == "preflight.json"]
    check(len(preflights) == 1, "runtime lock binds exact offline preflight aggregate")
    if len(preflights) == 1:
        preflight = read(preflights[0])
        check(
            set(preflight) == {"rule", "forced_observed_fallback"},
            "both offline rule and observed-only forced-fallback passes retained",
        )
        for arm in ("rule", "forced_observed_fallback"):
            metrics = preflight.get(arm, {})
            check(
                metrics.get("tasks") == metrics.get("completed_and_verified") == 24
                and metrics.get("required_operations")
                == metrics.get("successful_required_operations")
                == 70
                and metrics.get("successful_unwanted_writes") == 0
                and metrics.get("execution_errors") == 0,
                "offline exposed24/70-write regression gate passed: " + arm,
            )
    return {
        "failures": failures,
        "core_gate_passed": core_gate_passed,
        "argument_violation_trials": argument_violation_trials,
        "strict_quality_passed": strict_quality_passed,
        "strict_quality_failures_are_integrity_failures": False,
        "strict_quality_failure_details": [
            {
                key: row.get(key)
                for key in (
                    "task_id",
                    "arm",
                    "attempt",
                    "policy_completed",
                    "completed_and_verified",
                    "verifier_pass",
                    "authorization_blocked_attempts",
                    "prohibited_attempts",
                    "unresolved_invoices",
                    "successful_required_operations",
                    "required_operations",
                    "successful_unwanted_writes",
                    "fallback_error",
                )
            }
            for row in rows
            if row.get("completed_and_verified") is not True
        ],
        "native_trials_started": len(manifest.get("execution_order", [])),
        "retained_final_trials": len(rows),
        "max_job_seconds": manifest.get("max_job_seconds"),
        "total_execution_wall_seconds": manifest.get("run_total_wall_ms", 0) / 1000,
        "owned_environments_deleted": sum(
            row.get("cleanup") == "deleted-owned-environment" for row in rows
        ),
        "usage": {
            field: sum(row.get(field, 0) for row in rows)
            for field in ("llm_calls", "llm_input_tokens", "llm_output_tokens")
        },
        "native_evidence_files_verified": len(hashes),
        "native_evidence_sha256": hashes,
        "policy_failure_trials": sum(not row.get("completed_and_verified") for row in rows),
        "cohort_previously_exposed": True,
    }


def audit(root: Path, output: Path, original_freeze: Path) -> dict[str, Any]:
    run = root / "final-harbor"
    freeze_path = root / "final-freeze.json"
    manifest = read(run / "manifest.json")
    if manifest.get("complete") is not True:
        raise RuntimeError("Run is not complete; do not audit or modify live evaluation artifacts")
    freeze = read(freeze_path)
    cohort_path = run / "cohort-manifest.json"
    cohort = read(cohort_path)
    rows = records(run / "results.jsonl")
    failures = []
    checks = []
    hashes = {}

    def check(ok: bool, description: str) -> None:
        (checks if ok else failures).append(description)

    def guarded(description: str, fn: Any) -> Any:
        try:
            result = fn()
            checks.append(description)
            return result
        except Exception as exc:
            failures.append(f"{description}: {type(exc).__name__}: {exc}")
            return None

    report = guarded(
        "generalization_report independently recomputes every native/tool outcome",
        lambda: analyze(run),
    )
    check(manifest.get("error") is None, "runner completed without a terminal execution error")
    check(tuple(ARMS) == ("released", "llm"), "exact two-arm design")
    guarded(
        "all frozen runtime, cohort and artifact hashes match",
        lambda: check_final_freeze(freeze, cohort_path),
    )
    check(read(run / "freeze.json") == freeze, "saved freeze equals pre-run freeze")
    check(
        manifest.get("freeze_sha256") == digest(freeze_path), "manifest binds exact pre-run freeze"
    )
    check(manifest.get("arm_contracts") == freeze["arm_contracts"], "both arm contracts unchanged")
    access_path = Path(freeze["evaluation_access_path"])
    saved_access = run / "evaluation-access.json"
    access = guarded("external first-access marker readable", lambda: read(access_path))
    saved_claim = guarded("saved first-access marker readable", lambda: read(saved_access))
    check(
        access == saved_claim and manifest.get("evaluation_access_path") == str(access_path),
        "external and retained first-access marker agree",
    )
    if isinstance(access, dict):
        check(
            access.get("schema_version") == "prospective-cohort-access-v1"
            and access.get("cohort_manifest_sha256") == digest(cohort_path)
            and access.get("freeze_sha256") == digest(freeze_path)
            and access.get("output") == str(run.resolve())
            and access.get("claimed_before_model_initialization") is True,
            "first-access marker binds cohort, freeze, output and pre-initialization claim",
        )
        guarded(
            "external first-access marker hash matches final manifest",
            lambda: require_hash(access_path, manifest["evaluation_access_sha256"]),
        )
        guarded(
            "retained first-access marker hash matches final manifest",
            lambda: require_hash(saved_access, manifest["evaluation_access_sha256"]),
        )
    expected = {
        (task, arm, attempt) for task in freeze["task_ids"] for arm in ARMS for attempt in range(2)
    }
    indexed = {identity(row): row for row in rows}
    check(
        len(rows) == len(indexed) == manifest["planned_trials"] == len(expected) == 48,
        "48 unique final trials retained",
    )
    check(set(indexed) == expected, "exact task/arm/attempt paired coverage")
    order = manifest.get("execution_order", [])
    planned_order = []
    for attempt in range(2):
        for index, task_id in enumerate(freeze["task_ids"]):
            shift = (index + attempt) % len(ARMS)
            for arm in ARMS[shift:] + ARMS[:shift]:
                planned_order.append((task_id, arm, attempt))
    check(
        [identity(entry) for entry in order] == planned_order,
        "execution matches predeclared alternating arm/task/attempt order",
    )
    check(
        len(order) == 48 and [entry["position"] for entry in order] == list(range(48)),
        "48 contiguous execution positions",
    )
    check(
        all(
            identity(entry) == identity(rows[index])
            and rows[index].get("execution_position") == index
            for index, entry in enumerate(order)
            if index < len(rows)
        ),
        "aggregate rows match declared execution order",
    )
    check(
        manifest.get("training_steps") == freeze.get("training_steps") == 0,
        "zero training steps declared in frozen and final manifests",
    )
    check(
        manifest.get("calibration_fits") == freeze.get("calibration_fits") == 0,
        "zero calibration fits declared in frozen and final manifests",
    )
    check(
        manifest.get("device") == "mps"
        and manifest.get("max_tool_calls") == manifest.get("max_llm_calls") == 16
        and manifest.get("fallback_timeout") == 120
        and manifest.get("task_timeout") == 180
        and manifest.get("max_job_seconds") == freeze.get("max_job_seconds") == 2700,
        "local MPS execution and predeclared call/time budgets unchanged",
    )
    check(
        all(row.get("invoice_local_fallback") is True for row in rows),
        "invoice-local fallback used in every arm",
    )
    roots = Counter(
        cohort["tasks"][task_id]["composition_root"] for task_id in manifest["task_ids"]
    )
    check(
        len(manifest["task_ids"]) == 12 and len(roots) == 6 and set(roots.values()) == {2},
        "12 task variants in six roots with two variants each",
    )

    for task_id in manifest["task_ids"]:
        guarded(
            "unchanged native task " + task_id,
            lambda task_id=task_id: validate_task(cohort["tasks"][task_id]),
        )
    check(
        freeze.get("cohort_previously_exposed") is False
        and manifest.get("cohort_previously_exposed") is False
        and cohort.get("cohort_previously_exposed") is False,
        "prospective cohort consistently marked unexposed",
    )
    check(
        cohort.get("pre_freeze_policy_rollouts") == 0
        and cohort.get("pre_freeze_model_inferences") == 0
        and cohort.get("pre_freeze_authorization_parser_checks") == 0
        and cohort.get("pre_freeze_teacher_collections") == 0,
        "cohort provenance records zero pre-freeze policy, model, parser or teacher runs",
    )
    novelty = cohort.get("novelty_audit", {})
    check(
        novelty.get("unique_new_structures") == 6
        and all(
            novelty.get(field) == 0
            for field in (
                "root_collisions",
                "structural_collisions",
                "normalized_instruction_collisions",
            )
        ),
        "six distinct new structures with no prior root, structure or instruction collisions",
    )
    original = read(original_freeze)
    check(
        freeze.get("baseline_reference_freeze_path") == str(original_freeze.resolve())
        and freeze.get("baseline_reference_freeze_sha256") == digest(original_freeze),
        "prospective freeze binds the exact preceding generalization-v1 experiment",
    )
    check(
        freeze["checkpoints"]["released"] == original["checkpoints"]["released"],
        "released checkpoint identity matches previous frozen experiment",
    )
    lock_path = Path(freeze["runtime_lock_path"])
    lock = guarded("repaired runtime lock readable", lambda: read(lock_path))
    check(
        model_and_execution_artifacts(freeze) == model_and_execution_artifacts(original)
        and freeze["artifact_sha256"].get(str(original_freeze.resolve())) == digest(original_freeze)
        and freeze["artifact_sha256"].get(str(lock_path.resolve())) == digest(lock_path),
        "checkpoint, calibration, selection, config and wheel hashes unchanged; "
        "only reference-freeze and runtime-lock artifacts differ",
    )
    check(
        freeze["arm_contracts"]["released"]["threshold"]
        == original["arm_contracts"]["released"]["threshold"]
        == 0.7,
        "released routing threshold reused unchanged from generalization-v1",
    )
    check(
        cohort.get("cohort_id") == "billing-generalization-v2"
        and novelty.get("source_task_counts", {}).get("fallback-fresh") == 12
        and novelty.get("source_task_counts", {}).get("generalization-v1") == 12
        and novelty.get("unique_prior_tasks") == 436
        and novelty.get("unique_prior_roots") == 86,
        "fresh-v2 novelty includes every previous cohort including exposed generalization-v1",
    )
    if isinstance(lock, dict):
        check(
            lock.get("regression_passed") is True,
            "runtime lock declares core regression gate passed",
        )
        check(
            isinstance(lock.get("regression_gate_criteria"), (str, dict))
            and bool(lock["regression_gate_criteria"]),
            "runtime lock explicitly describes core acceptance criteria",
        )
        check(
            isinstance(lock.get("strict_quality_passed"), bool),
            "runtime lock separately declares strict quality outcome",
        )
        for source in final_source_paths():
            key = str(source.resolve())
            expected = original["source_sha256"].get(key)
            snapshot = original_freeze.parent / "final-harbor" / ("source-" + source.name)
            guarded(
                "retained generalization-v1 runtime snapshot: " + source.name,
                lambda snapshot=snapshot, expected=expected: require_hash(snapshot, expected),
            )
            check(
                lock.get("source_sha256", {}).get(key)
                == freeze["source_sha256"].get(key)
                == digest(source),
                "current runtime matches successful-regression lock: " + source.name,
            )
            if source.name not in {"harbor.py", "authorization.py"}:
                check(
                    digest(source) == expected,
                    "unrelated runtime/model source unchanged: " + source.name,
                )
        regression_evidence = lock.get("regression_evidence_sha256", {})
        check(bool(regression_evidence), "runtime lock binds nonempty regression evidence")
        for path, expected_hash in regression_evidence.items():
            guarded(
                "frozen regression evidence: " + path,
                lambda path=path, expected_hash=expected_hash: require_hash(
                    Path(path), expected_hash
                ),
            )
            if Path(path).exists():
                hashes[path] = digest(Path(path))
    else:
        regression_evidence = {}
    regression = guarded(
        "independent exposed-regression coverage, native files and offline gate audit",
        lambda: audit_regression(root, regression_evidence),
    )
    if isinstance(regression, dict):
        check(not regression["failures"], "exposed regression evidence passes independent checks")
        failures.extend("regression: " + issue for issue in regression["failures"])
        if isinstance(lock, dict):
            check(
                lock.get("strict_quality_passed") == regression["strict_quality_passed"],
                "runtime lock strict quality declaration matches retained paid outcomes",
            )
    for path, expected_hash in freeze["source_sha256"].items():
        saved = run / ("source-" + Path(path).name)
        guarded(
            "saved runtime source " + saved.name,
            lambda saved=saved, expected_hash=expected_hash: require_hash(saved, expected_hash),
        )
        if saved.exists():
            hashes[str(saved)] = digest(saved)
    cohort_evidence = freeze.get("cohort_evidence_sha256", {})
    check(
        {Path(path).name for path in cohort_evidence}
        == {"fixture-validation.json", "prototype-specs.json"}
        and len(cohort_evidence) == 2,
        "exact authored prototype and offline fixture evidence bound in freeze",
    )
    for path, expected_hash in cohort_evidence.items():
        for evidence in (Path(path), run / Path(path).name):
            guarded(
                "unchanged offline cohort evidence " + str(evidence),
                lambda evidence=evidence, expected_hash=expected_hash: require_hash(
                    evidence, expected_hash
                ),
            )
            if evidence.exists():
                hashes[str(evidence)] = digest(evidence)
    fixtures = guarded(
        "retained offline fixture evidence readable", lambda: read(run / "fixture-validation.json")
    )
    if isinstance(fixtures, list):
        check(
            len(fixtures) == 12
            and {entry.get("task_id") for entry in fixtures} == set(freeze["task_ids"])
            and all(
                entry.get("fixture_only") is True
                and entry.get("agent_or_policy_executed") is False
                and entry.get("expected_endpoint_accepted") is True
                and entry.get("missing_writes_rejected") is True
                for entry in fixtures
            ),
            "12 offline authored fixtures accepted and missing-write controls rejected",
        )

    trial_records = []
    native_hashes = {}
    for trial_path in sorted((run / "trials").glob("*/results.jsonl")):
        values = records(trial_path)
        check(len(values) == 1, "one record in " + str(trial_path))
        for row in values:
            trial_records.append(row)
            key = identity(row)
            aggregate = indexed.get(key, {})
            check(
                {k: v for k, v in aggregate.items() if k != "execution_position"} == row,
                "aggregate matches trial record " + str(key),
            )
            verifier = trial_path.parent / row["task_id"] / "verifier"
            for name, field in (("reward.json", "rewards"), ("diagnostics.json", "diagnostics")):
                path = verifier / name
                if path.exists():
                    check(
                        read(path) == row.get("verifier", {}).get(field),
                        "native " + name + " matches record " + str(key),
                    )
                    native_hashes[str(path)] = digest(path)
                else:
                    failures.append("missing native evidence: " + str(path))
            native_hashes[str(trial_path)] = digest(trial_path)
    check(
        len(trial_records) == len({identity(row) for row in trial_records}) == 48,
        "all 48 per-trial records retained without duplicates",
    )
    check(len(native_hashes) == 144, "144 exact per-trial record/reward/diagnostic files verified")

    budget_violations = [
        identity(row)
        for row in rows
        if not (
            isinstance(row.get("tool_call_count"), int)
            and 0 <= row["tool_call_count"] <= 16
            and isinstance(row.get("llm_calls"), int)
            and 0 <= row["llm_calls"] <= 16
        )
    ]
    check(not budget_violations, "shared16tool/16LLM bounds respected")
    successful_blocked_calls = [
        {"trial": identity(row), "observation": observation}
        for row in rows
        for observation in row.get("tool_observations", [])
        if (observation.get("authorization_blocked") or observation.get("invoice_scope_blocked"))
        and observation.get("result", {}).get("ok") is True
    ]
    check(not successful_blocked_calls, "zero successful authorization/scope-blocked calls")
    cleanup = [{"trial": identity(row), "cleanup": row.get("cleanup")} for row in rows]
    check(
        all(item["cleanup"] == "deleted-owned-environment" for item in cleanup),
        "every owned environment has recorded deletion evidence",
    )
    issues = [
        {
            "trial": identity(row),
            "error": row.get("error"),
            "verifier_error": row.get("verifier_error"),
            "fallback_error": row.get("fallback_error"),
            "policy_completed": row.get("policy_completed"),
            "completed_and_verified": row["completed_and_verified"],
            "tool_evidence_complete": row.get("tool_evidence_complete"),
            "indeterminate_tool_calls": row.get("indeterminate_tool_calls", []),
            "unresolved_invoices": row.get("unresolved_invoices", []),
            "missing_required_operation_details": row.get("missing_required_operation_details", []),
        }
        for row in rows
        if not row["completed_and_verified"]
        or row.get("error")
        or row.get("verifier_error")
        or row.get("fallback_error")
        or row.get("tool_evidence_complete") is False
        or row.get("indeterminate_tool_calls")
    ]
    proposals = [
        decision.get("proposal", {})
        for row in rows
        if row["arm"].startswith("released")
        for decision in row.get("decisions", [])
    ]
    known_usage = {}
    for field in ("llm_calls", "llm_input_tokens", "llm_output_tokens"):
        measured = [row.get(field) for row in rows]
        known_usage[field] = {
            "known_total": sum(value for value in measured if isinstance(value, int)),
            "trials_with_known_usage": sum(isinstance(value, int) for value in measured),
            "trials_missing_usage": sum(not isinstance(value, int) for value in measured),
        }
    accounting = {
        "native_trials_started": len(order),
        "fresh_run_max_job_seconds": 2700,
        "exposed_regression_max_job_seconds": 900,
        "regression_accounting": regression,
        "retained_final_trials": len(rows),
        "usage": known_usage,
        "total_execution_wall_seconds": manifest.get("run_total_wall_ms", 0) / 1000,
        "owned_environments_deleted": sum(
            item["cleanup"] == "deleted-owned-environment" for item in cleanup
        ),
        "cleanup_evidence": cleanup,
        "training_steps": manifest["training_steps"],
        "calibration_fits": manifest["calibration_fits"],
        "device": manifest.get("device"),
        "region": "us-west-2",
        "new_aws_instances": 0,
        "gpu_instance_occupancy_seconds": 0,
        "execution_error_trials": sum(bool(row.get("error")) for row in rows),
        "verifier_error_trials": sum(bool(row.get("verifier_error")) for row in rows),
        "incomplete_tool_evidence_trials": sum(
            row.get("tool_evidence_complete") is False for row in rows
        ),
        "indeterminate_tool_calls": [
            {"trial": identity(row), "call": call}
            for row in rows
            for call in row.get("indeterminate_tool_calls", [])
        ],
        "issues": issues,
        "notes": [
            "All completed trial records are included, including failed policies "
            "and unresolved work.",
            "Usage is returned provider accounting and may omit failed or in-flight usage; "
            "no dollar total is inferred.",
            "Environment deletion is audited from retained runner evidence; "
            "this script does not query Docker.",
            "No new infrastructure and zero GPU instance occupancy describe this local "
            "MPS/Bedrock experiment; this script does not query AWS billing.",
        ],
    }
    for path in (
        freeze_path,
        original_freeze,
        run / "freeze.json",
        cohort_path,
        run / "manifest.json",
        run / "results.jsonl",
        access_path,
        saved_access,
        lock_path,
        Path(__file__),
    ):
        if path.exists():
            hashes[str(path)] = digest(path)
    result = {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "native_evidence_files_verified": len(native_hashes),
        "native_evidence_sha256": native_hashes,
        "source_and_input_sha256": hashes,
        "budget_violations": budget_violations,
        "successful_blocked_calls": successful_blocked_calls,
        "model_proposal_count": len(proposals),
        "max_model_input_tokens": max((p.get("input_tokens", 0) for p in proposals), default=0),
        "model_proposal_errors": [p for p in proposals if p.get("error")],
        "policy_failure_trials": len([row for row in rows if not row["completed_and_verified"]]),
        "policy_failures_are_integrity_failures": False,
        "regression_audit": regression,
    }
    output.mkdir(parents=True, exist_ok=True)
    write(output / "audit.json", result)
    write(output / "resource-accounting.json", accounting)
    if report is not None:
        write(output / "audit-recomputed-metrics.json", report)
    return result


def require_hash(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError("hash mismatch: " + str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(".artifacts/decision-model-generalization-v2")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/decision-model-generalization-v2")
    )
    parser.add_argument(
        "--original-freeze",
        type=Path,
        default=Path(".artifacts/decision-model-generalization/final-freeze.json"),
    )
    args = parser.parse_args()
    result = audit(args.root, args.output, args.original_freeze)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "passed",
                    "failures",
                    "native_evidence_files_verified",
                    "policy_failure_trials",
                )
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
