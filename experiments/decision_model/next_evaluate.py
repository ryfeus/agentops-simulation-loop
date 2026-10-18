"""Whole-task checkpoint selection and frozen final comparisons, with real billing execution.

Development uses the rendered task's SQLite seed and exact native verifier without
Docker. Final comparisons use native Harbor. Neither path supplies hidden expected
operations, initial database rows, or verifier labels to the predictor.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_tasks(
    manifest_path: Path, split: str, limit: int, tasks_root: Path | None = None
) -> list[dict[str, Any]]:
    """Identity-only family balancing; no policy/model evaluation during selection."""
    manifest = json.loads(Path(manifest_path).read_text())
    task_map = manifest["tasks"]
    values = list(task_map.values()) if isinstance(task_map, dict) else list(task_map)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in values:
        if value["split"] != split:
            continue
        item = dict(value)
        if tasks_root is not None:
            item["task_path"] = str(Path(tasks_root) / item["task_id"])
        family = item.get("composition_root") or item.get("family_id") or item["group_id"]
        groups[family].append(item)
    queues = [sorted(groups[key], key=lambda row: row["task_id"]) for key in sorted(groups)]
    selected = []
    while any(queues) and len(selected) < limit:
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.pop(0))
    if not selected:
        raise ValueError(f"No tasks selected for split {split}")
    return selected


def validate_task(task: dict[str, Any]) -> None:
    from agentops_demo.taskify.integrity import harbor_task_sha256, scenario_sha256
    from agentops_demo.validation.scenario import load_scenario

    path = Path(task["task_path"])
    if harbor_task_sha256(path) != task["execution_task_sha256"]:
        raise ValueError(f"Native task contents changed: {task['task_id']}")
    if scenario_sha256(load_scenario(path / "scenario.yaml")) != task["scenario_sha256"]:
        raise ValueError(f"Scenario changed: {task['task_id']}")


class LocalSQLite:
    """The same BillingService and native verifier, with an isolated local database."""

    def __init__(self, task_dir: Path, output: Path):
        self.task_dir, self.output = task_dir, output
        self.database = output / "billing.db"
        self.trajectory = output / "agent-run.json"
        self.calls: list[dict[str, Any]] = []
        self.task = SimpleNamespace(instruction=(task_dir / "instruction.md").read_text())

    async def start(self) -> None:
        from agentops_demo.billing.service import BillingService
        from agentops_demo.billing.sqlite_repository import SQLiteBillingRepository

        self.output.mkdir(parents=True, exist_ok=False)
        with sqlite3.connect(self.database) as connection:
            connection.executescript((self.task_dir / "environment/seed.sql").read_text())
        write(self.trajectory, {"tool_calls": []})
        self.service = BillingService(
            SQLiteBillingRepository(self.database), policy_mode="enforced"
        )

    async def invoke(self, name: str, arguments: dict[str, str]) -> dict[str, Any]:
        from agentops_demo.billing.errors import BillingDomainError
        from experiments.decision_model.harbor import read_escalation_history

        if name not in {"get_invoice", "refund_invoice", "escalate_dispute"}:
            raise ValueError("Unsupported billing tool")
        self.calls.append({"name": name, "arguments": dict(arguments)})
        write(self.trajectory, {"tool_calls": self.calls})
        try:
            value = await getattr(self.service, name)(**arguments)
            if value is None:
                return {
                    "ok": False,
                    "error_type": "InvoiceNotFoundError",
                    "error": "invoice not found",
                }
            key = {
                "get_invoice": "invoice",
                "refund_invoice": "refund",
                "escalate_dispute": "escalation",
            }[name]
            result = {"ok": True, key: value.model_dump(mode="json")}
            if name == "get_invoice":
                result["escalation_history"] = read_escalation_history(
                    self.database, arguments["invoice_id"]
                )
            return result
        except BillingDomainError as exc:
            return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}

    async def verify(self) -> dict[str, Any]:
        started = time.perf_counter()
        spec = importlib.util.spec_from_file_location(
            "native_verifier_" + uuid.uuid4().hex, self.task_dir / "tests/verify.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load exact native task verifier")
        verifier = importlib.util.module_from_spec(spec)
        # Execute the exact verifier source without creating bytecode in the frozen
        # task tree. SourceFileLoader.exec_module would mutate its all-files hash.
        source = (self.task_dir / "tests/verify.py").read_bytes()
        exec(compile(source, str(self.task_dir / "tests/verify.py"), "exec"), verifier.__dict__)
        verifier.AGENT_RESULT_PATH = self.trajectory
        verifier.CONTAINER_SCENARIO_PATH = self.task_dir / "tests/scenario.json"
        rewards, diagnostics = verifier.evaluate_database(self.database)
        write(self.output / "reward.json", rewards)
        write(self.output / "diagnostics.json", diagnostics)
        return {
            "rewards": rewards,
            "diagnostics": diagnostics,
            "verifier_ms": (time.perf_counter() - started) * 1000,
        }

    async def close(self) -> None:
        # Repository connections close per operation. Retain local DBs as reviewable evidence.
        return None


def task_outcomes(result: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Hidden intent is consulted only after the policy and verifier have finished."""
    expected = task["expected_operations"]
    successes = set()
    unwanted = 0
    for item in result.get("tool_observations", []):
        if item["name"] not in {"refund_invoice", "escalate_dispute"}:
            continue
        target = item["arguments"].get("invoice_id")
        if item["result"].get("ok") is True:
            if expected.get(target) == item["name"]:
                successes.add(target)
            else:
                unwanted += 1
    rewards = result.get("verifier", {}).get("rewards", {})
    verified = bool(rewards) and rewards.get("reward") == 1.0
    missing = [
        {"invoice_id": target, "operation": operation}
        for target, operation in expected.items()
        if target not in successes
    ]
    return {
        "verifier_pass": verified,
        "completed_and_verified": verified
        and result.get("policy_completed") is True
        and not result.get("error"),
        "successful_unwanted_writes": unwanted,
        "missing_required_operations": len(missing),
        "missing_required_operation_details": missing,
        "successful_required_operations": len(successes),
        "required_operations": len(expected),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot score an empty evaluation")
    proposals = sum(len(row.get("decisions", [])) for row in records)
    prohibited = sum(row.get("prohibited_proposals", 0) for row in records)
    accepted = [
        decision
        for row in records
        for decision in row.get("decisions", [])
        if not decision.get("fallback_reason")
    ]
    inference_errors = sum(
        bool(decision.get("proposal", {}).get("error"))
        or decision.get("fallback_reason") in {"model_error", "timeout"}
        for row in records
        for decision in row.get("decisions", [])
    )
    result = {
        "tasks": len(records),
        "completed_and_verified": sum(row["completed_and_verified"] for row in records),
        "verifier_passes": sum(row["verifier_pass"] for row in records),
        "successful_unwanted_writes": sum(row["successful_unwanted_writes"] for row in records),
        "missing_required_operations": sum(row["missing_required_operations"] for row in records),
        "successful_required_operations": sum(
            row["successful_required_operations"] for row in records
        ),
        "required_operations": sum(row["required_operations"] for row in records),
        "raw_prohibited_proposals": prohibited,
        "proposal_count": proposals,
        "raw_prohibited_proposal_rate": prohibited / proposals if proposals else 0.0,
        "accepted_prohibited_proposals": sum(bool(d.get("prohibited_proposal")) for d in accepted),
        "accepted_proposal_count": len(accepted),
        "authorization_blocked_attempts": sum(
            row.get("authorization_blocked_attempts", 0) for row in records
        ),
        "fallback_tasks": sum(bool(row.get("fallback_count")) for row in records),
        "fallback_continuations": sum(row.get("fallback_count", 0) for row in records),
        "fallback_unresolved": sum(bool(row.get("fallback_error")) for row in records),
        "invoice_scope_blocked_attempts": sum(
            row.get("invoice_scope_blocked_attempts", 0) for row in records
        ),
        "unresolved_invoice_count": sum(len(row.get("unresolved_invoices", [])) for row in records),
        "incomplete_tool_evidence_tasks": sum(
            row.get("tool_evidence_complete") is False for row in records
        ),
        "indeterminate_tool_call_count": sum(
            len(row.get("indeterminate_tool_calls", [])) for row in records
        ),
        "indeterminate_write_call_count": sum(
            call.get("name") in {"refund_invoice", "escalate_dispute"}
            for row in records
            for call in row.get("indeterminate_tool_calls", [])
        ),
        "execution_errors": sum(bool(row.get("error")) for row in records),
        "inference_errors": inference_errors,
    }
    for field in ("llm_calls", "llm_input_tokens", "llm_output_tokens"):
        values = [row.get(field) for row in records]
        result[field] = sum(values) if all(isinstance(value, int) for value in values) else None
    result["selection_key"] = [
        -result["successful_unwanted_writes"],
        result["completed_and_verified"],
        -result["missing_required_operations"],
        -result["raw_prohibited_proposal_rate"],
    ]
    result["selection_eligible"] = (
        result["execution_errors"] == 0
        and inference_errors == 0
        and result["incomplete_tool_evidence_tasks"] == 0
    )
    return result


async def evaluate_tasks(
    predict: Any,
    tasks: list[dict[str, Any]],
    output: Path,
    *,
    threshold: float = 0,
    fallback: Any = None,
    backend: str = "sqlite",
    wheel: Path | None = None,
    max_tool_calls: int = 16,
    decision_timeout: float = 30,
    task_timeout: float = 180,
    allow_test: bool = False,
    arm: str = "candidate",
    attempt: int = 0,
    invoice_local_fallback: bool = False,
    episode_runner: Any = None,
) -> dict[str, Any]:
    """Predictor is loaded by the caller and receives only actual observed state.

    Development defaults to no fallback: defer remains explicitly unresolved. A
    final comparison must opt in to test access and provide its frozen selection.
    """
    from agentops_demo.validation.scenario import load_scenario
    from experiments.decision_model.authorization import scoped_rule_action
    from experiments.decision_model.harbor import (
        PROMPT_SUFFIX,
        NativeHarbor,
        execution_safety,
        run_episode,
    )

    output = Path(output)
    if not tasks or len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Tasks must be nonempty and unique")
    if any(task.get("split") == "test" for task in tasks) and not allow_test:
        raise ValueError("Development evaluation may not inspect test outcomes")
    if backend not in {"sqlite", "harbor"} or (backend == "harbor" and wheel is None):
        raise ValueError("Choose SQLite, or native Harbor with a wheel")
    for task in tasks:
        validate_task(task)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {
        "schema_version": "whole-task-decider-evaluation-v1",
        "backend": backend,
        "arm": arm,
        "attempt": attempt,
        "escalation_history": True,
        "invoice_authorization": True,
        "invoice_local_fallback": invoice_local_fallback,
        "max_llm_calls": 16 if invoice_local_fallback else None,
        "fallback_timeout": 120,
        "threshold": threshold,
        "fallback_enabled": fallback is not None,
        "max_tool_calls": max_tool_calls,
        "decision_timeout": decision_timeout,
        "task_timeout": task_timeout,
        "task_ids": [task["task_id"] for task in tasks],
        "task_hashes": {task["task_id"]: task["execution_task_sha256"] for task in tasks},
        "evaluator_sha256": digest(Path(__file__)),
        "complete": False,
        "scope": "Actual local billing service and exact native verifier; no Docker timing"
        if backend == "sqlite"
        else "Native Harbor Docker execution and verifier",
    }
    write(output / "manifest.json", manifest)
    records = []
    for task in tasks:
        validate_task(task)
        path = Path(task["task_path"])
        scenario = load_scenario(path / "scenario.yaml")
        if scenario.benchmark is None:
            raise ValueError("Task lacks harness target identities")
        targets = scenario.benchmark.trajectory.required_inspections
        env = (
            LocalSQLite(path, output / task["task_id"])
            if backend == "sqlite"
            else NativeHarbor(
                path, wheel, output / task["task_id"], "enforced", observe_escalations=True
            )
        )
        predictor = predict
        if predictor is None:

            def predictor(state: dict[str, Any], targets: list[str] = targets) -> dict[str, Any]:
                return {
                    "action": scoped_rule_action(state, targets, escalation_history=True),
                    "confidence": 1.0,
                }

        result = {
            "task_id": task["task_id"],
            "split": task.get("split"),
            "arm": arm,
            "attempt": attempt,
            "invoice_local_fallback": invoice_local_fallback,
        }
        task_started = time.perf_counter()
        episode_sink: dict[str, Any] = {}
        env_started = False
        episode_started = False
        cancelled = False

        async def run(
            env: Any = env,
            targets: list[str] = targets,
            predictor: Any = predictor,
            result: dict[str, Any] = result,
            scenario: Any = scenario,
            episode_sink: dict[str, Any] = episode_sink,
        ) -> None:
            nonlocal env_started, episode_started
            await env.start()
            env_started = True
            episode_started = True
            episode = await (episode_runner or run_episode)(
                request=env.task.instruction + PROMPT_SUFFIX,
                targets=targets,
                invoke=env.invoke,
                predict=predictor,
                threshold=threshold,
                timeout=decision_timeout,
                fallback=fallback,
                max_tool_calls=max_tool_calls,
                escalation_history=True,
                invoice_authorization=True,
                invoice_local_fallback=invoice_local_fallback,
                max_llm_calls=16,
                fallback_timeout=120,
                evidence_sink=episode_sink,
            )
            result.update(episode)

        try:
            await asyncio.wait_for(run(), timeout=task_timeout)
        except asyncio.CancelledError:
            result["error"] = "CancelledError: evaluation cancelled"
            cancelled = True
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            # A cancelled coroutine cannot return its local observations/callback usage.
            # The episode's cancellation snapshot keeps completed writes and known usage.
            result.update(episode_sink)
            if result.get("error"):
                result["policy_completed"] = False
                if episode_started and not episode_sink:
                    result["tool_evidence_complete"] = False
            if "tool_observations" in result:
                result.update(
                    execution_safety(result, scenario.benchmark.mutation_policy.allowed_targets)
                )
            if env_started:
                try:
                    result["verifier"] = await asyncio.wait_for(env.verify(), timeout=30)
                except asyncio.CancelledError:
                    cancelled = True
                    result["error"] = "CancelledError: native verification cancelled"
                    result["policy_completed"] = False
                except Exception as exc:
                    result["verifier_error"] = f"{type(exc).__name__}: {exc}"
                    result.setdefault(
                        "error", "Native verifier failed: " + result["verifier_error"]
                    )
            try:
                await asyncio.wait_for(env.close(), timeout=30)
                result["cleanup"] = (
                    "local-evidence-retained"
                    if backend == "sqlite"
                    else "deleted-owned-environment"
                )
            except Exception as exc:
                result["cleanup"] = f"failed: {exc}"
        result["total_wall_ms"] = (time.perf_counter() - task_started) * 1000
        result.update(task_outcomes(result, task))
        records.append(result)
        with (output / "results.jsonl").open("a") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        if cancelled:
            manifest.update(error=result["error"], complete=False)
            write(output / "manifest.json", manifest)
            raise asyncio.CancelledError("Evaluation cancelled; partial evidence retained")
        if result["cleanup"].startswith("failed:"):
            raise RuntimeError("Environment cleanup failed; stopped before further trials")
    metrics = summarize(records)
    metrics["records_path"] = str(output / "results.jsonl")
    metrics["run_total_wall_ms"] = (time.perf_counter() - started) * 1000
    write(output / "metrics.json", metrics)
    manifest.update(complete=True, metrics=metrics)
    write(output / "manifest.json", manifest)
    return metrics


async def evaluate_candidate(
    manifest_path: Path,
    predict: Any,
    output: Path,
    *,
    split: str = "dev",
    limit: int = 8,
    threshold: float = 0,
    fallback: Any = None,
    tasks_root: Path | None = None,
    invoice_local_fallback: bool = False,
) -> dict[str, Any]:
    if split not in {"train", "dev", "calibration"}:
        raise ValueError(
            "Development/routing evaluation accepts train/dev/calibration only; "
            "test remains reserved"
        )
    maximum = 48 if split == "train" else 24
    if not 1 <= limit <= maximum:
        raise ValueError(f"Candidate evaluation is bounded to 1-{maximum} tasks")
    tasks = load_tasks(Path(manifest_path), split, limit, tasks_root)
    return await evaluate_tasks(
        predict,
        tasks,
        Path(output),
        threshold=threshold,
        fallback=fallback,
        invoice_local_fallback=invoice_local_fallback,
    )


def final_source_paths() -> list[Path]:
    return [
        Path(__file__).with_name(name).resolve()
        for name in (
            "next_evaluate.py",
            "harbor.py",
            "authorization.py",
            "data.py",
            "history.py",
            "model.py",
        )
    ]


def build_final_freeze(
    manifest: Path,
    *,
    released: Path,
    tuned: Path,
    released_calibration: Path,
    tuned_calibration: Path,
    fallback_config: Path,
    wheel: Path,
    selection_manifest: Path,
    invoice_local_fallback: bool = False,
    cohort_previously_exposed: bool = False,
    selected_checkpoint_role: str = "unspecified",
) -> dict[str, Any]:
    """Prepare after selection/calibration; reads identities, never test outcomes."""
    from experiments.decision_model.model import checkpoint_identity

    if json.loads(selection_manifest.read_text()).get("selection_complete") is not True:
        raise ValueError("Selection evidence is not marked complete")
    if selected_checkpoint_role not in {"unspecified", "trained", "released_baseline"}:
        raise ValueError("Unknown selected checkpoint role")
    artifacts = [
        released_calibration,
        tuned_calibration,
        fallback_config,
        wheel,
        selection_manifest,
    ]
    checkpoints = {}
    for arm, path in (("released", released), ("tuned", tuned)):
        identity = checkpoint_identity(path)
        if not identity["files_sha256"]:
            raise ValueError(f"Checkpoint has no identity files: {arm}")
        checkpoints[arm] = identity
        artifacts.extend(Path(identity["path"]) / name for name in identity["files_sha256"])
    return {
        "schema_version": "whole-task-final-freeze-v1",
        "selection_complete": True,
        "invoice_local_fallback": invoice_local_fallback,
        "cohort_previously_exposed": cohort_previously_exposed,
        "selected_checkpoint_role": selected_checkpoint_role,
        "cohort_manifest_sha256": digest(manifest),
        "source_sha256": {str(path): digest(path) for path in final_source_paths()},
        "artifact_sha256": {str(Path(path).resolve()): digest(Path(path)) for path in artifacts},
        "checkpoints": checkpoints,
        "selection_manifest": str(selection_manifest.resolve()),
    }


def check_final_freeze(freeze: dict[str, Any], manifest: Path) -> None:
    if freeze.get("selection_complete") is not True:
        raise ValueError("Final test access requires completed development selection")
    selection = Path(freeze["selection_manifest"])
    if str(selection.resolve()) not in freeze["artifact_sha256"]:
        raise ValueError("Selection evidence is absent from frozen artifacts")
    if json.loads(selection.read_text()).get("selection_complete") is not True:
        raise ValueError("Selection evidence is not marked complete")
    if digest(manifest) != freeze["cohort_manifest_sha256"]:
        raise ValueError("Frozen cohort changed")
    sources = {str(Path(path).resolve()): value for path, value in freeze["source_sha256"].items()}
    if not {str(path) for path in final_source_paths()} <= set(sources):
        raise ValueError("Final freeze omits a runtime source")
    for path, expected in {**sources, **freeze["artifact_sha256"]}.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen source or artifact changed: {path}")


def model_predictor(model: Any) -> Any:
    from experiments.decision_model.data import INSTRUCTIONS, OPTIONS, serialize_state

    def predict(state: dict[str, Any]) -> dict[str, Any]:
        return model.predict(
            {
                "kind": "choice",
                "state": serialize_state(state),
                "instructions": INSTRUCTIONS,
                "options": OPTIONS,
                "label": 0,
            }
        )

    return predict


async def final_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """One frozen four-arm native experiment; no fitting, selection, or quality retries."""
    from experiments.decision_model.harbor import llm_fallback
    from experiments.decision_model.model import DecisionModel, checkpoint_identity

    started = time.perf_counter()
    if not 1 <= args.limit <= 12 or args.repeats != 2:
        raise ValueError("Final comparison requires two attempts on at most12 tasks")
    if not 60 <= args.max_job_seconds <= 3600:
        raise ValueError("Final job budget must be60-3600 seconds")
    freeze = json.loads(args.freeze.read_text())
    check_final_freeze(freeze, args.manifest)
    options = {
        "invoice_local_fallback": getattr(args, "invoice_local_fallback", False),
        "cohort_previously_exposed": getattr(args, "cohort_previously_exposed", False),
        "selected_checkpoint_role": getattr(args, "selected_checkpoint_role", "unspecified"),
    }
    for key, value in options.items():
        default = "unspecified" if key == "selected_checkpoint_role" else False
        if freeze.get(key, default) != value:
            raise ValueError(f"Final command differs from frozen evaluation option: {key}")
    required = [args.wheel, args.fallback_config, args.released_calibration, args.tuned_calibration]
    if any(str(path.resolve()) not in freeze["artifact_sha256"] for path in required):
        raise ValueError("Final command uses an unfrozen artifact")
    for arm in ("released", "tuned"):
        if checkpoint_identity(getattr(args, arm)) != freeze["checkpoints"][arm]:
            raise ValueError(f"Final {arm} checkpoint differs from selection freeze")
    tasks = load_tasks(args.manifest, "test", args.limit, args.tasks_root)
    for task in tasks:
        validate_task(task)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "freeze.json", freeze)
    (args.output / "cohort-manifest.json").write_bytes(args.manifest.read_bytes())
    for path in final_source_paths():
        (args.output / ("source-" + path.name)).write_bytes(path.read_bytes())
    os.environ.update(
        {
            "AWS_REGION": "us-west-2",
            "AWS_DEFAULT_REGION": "us-west-2",
            "BILLING_MODEL_TEMPERATURE": "0",
            "BILLING_MODEL_MAX_TOKENS": "512",
            "BILLING_MODEL_TOTAL_ATTEMPTS": "1",
        }
    )
    predictors: dict[str, Any] = {
        "rule": None,
        "llm": lambda state: {"action": "defer", "confidence": 1.0},
    }
    thresholds = {"rule": 0.0, "llm": 0.0}
    model_load_ms = {}
    models = {}
    for arm in ("released", "tuned"):
        loaded = time.perf_counter()
        calibration_path = getattr(args, arm + "_calibration")
        calibration = json.loads(calibration_path.read_text())
        thresholds[arm] = float(calibration["threshold"])
        if not 0 <= thresholds[arm] <= 1.000001:
            raise ValueError("Invalid frozen routing threshold")
        models[arm] = DecisionModel(
            getattr(args, arm), device=args.device, calibration=calibration_path
        )
        predictors[arm] = model_predictor(models[arm])
        model_load_ms[arm] = (time.perf_counter() - loaded) * 1000
    fallback_started = time.perf_counter()
    fallback = llm_fallback(
        args.fallback_config, invoice_local_fallback=options["invoice_local_fallback"]
    )
    manifest = {
        "schema_version": "whole-task-four-arm-final-v1",
        "complete": False,
        "task_ids": [task["task_id"] for task in tasks],
        "repeats": 2,
        "planned_trials": len(tasks) * 8,
        "max_job_seconds": args.max_job_seconds,
        "backend": "native-harbor",
        "escalation_history": True,
        "invoice_authorization": True,
        **options,
        "max_tool_calls": 16,
        "max_llm_calls": 16 if options["invoice_local_fallback"] else None,
        "fallback_timeout": 120,
        "thresholds": thresholds,
        "device": args.device,
        "model_load_ms": model_load_ms,
        "fallback_load_ms": (time.perf_counter() - fallback_started) * 1000,
        "freeze_sha256": digest(args.freeze),
        "cohort_manifest_sha256": digest(args.manifest),
        "execution_order": [],
        "error": None,
        "limitations": [
            "Two attempts per task are correlated repetitions, not independent families.",
            "Standalone development routing differs from final hybrid fallback execution.",
            "Initial observed inputs and tool budget are common; "
            "later histories depend on actions.",
        ],
    }
    records: dict[str, list[dict[str, Any]]] = {arm: [] for arm in predictors}
    write(args.output / "manifest.json", manifest)
    order = ["rule", "released", "tuned", "llm"]
    try:
        for attempt in range(2):
            for index, task in enumerate(tasks):
                shift = (index + attempt) % len(order)
                for arm in order[shift:] + order[:shift]:
                    remaining = args.max_job_seconds - (time.perf_counter() - started)
                    if remaining < 65:
                        raise TimeoutError("Insufficient budget for another isolated trial")
                    check_final_freeze(freeze, args.manifest)
                    position = len(manifest["execution_order"])
                    manifest["execution_order"].append(
                        {
                            "task_id": task["task_id"],
                            "arm": arm,
                            "attempt": attempt,
                            "position": position,
                        }
                    )
                    write(args.output / "manifest.json", manifest)
                    output = (
                        args.output / "trials" / f"{position:03d}-{arm}-{attempt}-{task['task_id']}"
                    )
                    metrics = await evaluate_tasks(
                        predictors[arm],
                        [task],
                        output,
                        threshold=thresholds[arm],
                        fallback=fallback,
                        backend="harbor",
                        wheel=args.wheel,
                        max_tool_calls=16,
                        decision_timeout=30,
                        task_timeout=min(180, remaining - 60),
                        allow_test=True,
                        arm=arm,
                        attempt=attempt,
                        invoice_local_fallback=options["invoice_local_fallback"],
                    )
                    record = json.loads(Path(metrics["records_path"]).read_text())
                    record["execution_position"] = position
                    records[arm].append(record)
                    with (args.output / "results.jsonl").open("a") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(
                        json.dumps(
                            {
                                "position": position,
                                "arm": arm,
                                "task_id": task["task_id"],
                                "attempt": attempt,
                                "completed_and_verified": record["completed_and_verified"],
                                "llm_calls": record.get("llm_calls"),
                            }
                        ),
                        flush=True,
                    )
        manifest["complete"] = True
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["run_total_wall_ms"] = (time.perf_counter() - started) * 1000
        manifest["arms"] = {arm: summarize(rows) for arm, rows in records.items() if rows}
        write(args.output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--released", type=Path)
    parser.add_argument("--tuned", type=Path)
    parser.add_argument("--released-calibration", type=Path)
    parser.add_argument("--tuned-calibration", type=Path)
    parser.add_argument("--fallback-config", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-job-seconds", type=int, default=1800)
    parser.add_argument("--invoice-local-fallback", action="store_true")
    parser.add_argument("--cohort-previously-exposed", action="store_true")
    parser.add_argument(
        "--selected-checkpoint-role",
        choices=("unspecified", "trained", "released_baseline"),
        default="unspecified",
    )
    args = parser.parse_args()
    if args.final:
        if any(
            getattr(args, name) is None
            for name in (
                "freeze",
                "released",
                "tuned",
                "released_calibration",
                "tuned_calibration",
                "fallback_config",
                "wheel",
            )
        ):
            parser.error(
                "Final comparison requires freeze, two calibrated checkpoints, fallback, wheel"
            )
        result = asyncio.run(final_comparison(args))
    else:
        result = asyncio.run(
            evaluate_candidate(
                args.manifest,
                None,
                args.output,
                limit=args.limit,
                tasks_root=args.tasks_root,
                invoice_local_fallback=args.invoice_local_fallback,
            )
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
