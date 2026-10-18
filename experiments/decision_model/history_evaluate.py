"""Bounded paired history OFF/ON evaluation; real Docker and paid Bedrock calls."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

from scripts.aws_context import expected_account_id


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def task_digest(path: Path) -> str:
    from agentops_demo.taskify.integrity import harbor_task_sha256

    return harbor_task_sha256(path)


def task_order(tasks: dict[str, dict[str, Any]]) -> list[str]:
    """Interleave authored families so run timing does not line up with family blocks."""
    groups: dict[str, list[str]] = defaultdict(list)
    for key, task in tasks.items():
        groups[task.get("prototype_id") or task["group_id"]].append(key)
    queues = [sorted(groups[key]) for key in sorted(groups)]
    result = []
    while any(queues):
        for queue in queues:
            if queue:
                result.append(queue.pop(0))
    return result


async def evaluate(args: argparse.Namespace) -> None:
    from agentops_demo.taskify.integrity import scenario_sha256
    from agentops_demo.validation.scenario import load_scenario
    from experiments.decision_model import data, history
    from experiments.decision_model.harbor import (
        PROMPT_SUFFIX,
        NativeHarbor,
        execution_safety,
        llm_fallback,
        run_episode,
    )

    started = time.perf_counter()
    invoice_authorization = getattr(args, "invoice_authorization", False)
    cohort = json.loads(args.cohort.read_text())
    freeze_path = getattr(args, "policy_freeze", None) or args.cohort.parent / "policy-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["cohort_manifest_sha256"] != digest(args.cohort):
        raise ValueError("Cohort differs from the pre-evaluation policy freeze")
    for source, expected in freeze["source_sha256"].items():
        if digest(Path(source)) != expected:
            raise ValueError(f"Policy source differs from the frozen revision: {source}")
    authorization = None
    if invoice_authorization:
        from experiments.decision_model import authorization

        authorization_path = Path(authorization.__file__).resolve()
        if authorization_path not in {Path(source).resolve() for source in freeze["source_sha256"]}:
            raise ValueError("Invoice authorization source is absent from the policy freeze")
    tasks = cohort["tasks"]
    if not isinstance(tasks, dict) or not 1 <= len(tasks) <= 24:
        raise ValueError("Frozen paired cohort must contain 1-24 unique tasks")
    if args.output.exists():
        raise FileExistsError("Choose a fresh output directory; prior evidence is preserved")
    if not args.wheel.is_file() or not args.fallback_config.is_file():
        raise FileNotFoundError("Existing wheel and real Bedrock fallback configuration required")
    order = task_order(tasks)
    task_paths = {key: Path(tasks[key]["task_path"]).resolve() for key in tasks}
    task_hashes = {key: task_digest(path) for key, path in task_paths.items()}
    scenario_hashes = {
        key: scenario_sha256(load_scenario(path / "scenario.yaml"))
        for key, path in task_paths.items()
    }
    scenario_raw_hashes = {key: digest(path / "scenario.yaml") for key, path in task_paths.items()}
    for key in tasks:
        if tasks[key].get("scenario_sha256") != scenario_hashes[key]:
            raise ValueError(f"Cohort scenario changed before evaluation: {key}")
        if tasks[key].get("execution_task_sha256") not in (None, task_hashes[key]):
            raise ValueError(f"Rendered task changed before evaluation: {key}")
    source_paths = {
        name: Path(__file__).with_name(name)
        for name in ("harbor.py", "data.py", "history.py", "history_evaluate.py")
    }
    if invoice_authorization:
        source_paths["authorization.py"] = authorization_path
    source_hashes = {name: digest(path) for name, path in source_paths.items()}
    args.output.mkdir(parents=True)
    for name, path in source_paths.items():
        (args.output / ("source-" + name)).write_bytes(path.read_bytes())
    (args.output / "cohort-manifest.json").write_bytes(args.cohort.read_bytes())
    (args.output / "policy-freeze.json").write_bytes(freeze_path.read_bytes())
    cohort_hash, wheel_hash = digest(args.cohort), digest(args.wheel)
    inference = {
        "BILLING_MODEL_TEMPERATURE": "0",
        "BILLING_MODEL_MAX_TOKENS": "512",
        "BILLING_MODEL_TOTAL_ATTEMPTS": "1",
        "AWS_REGION": "us-west-2",
        "AWS_DEFAULT_REGION": "us-west-2",
    }
    os.environ.update(inference)
    fallback_started = time.perf_counter()
    fallback = llm_fallback(args.fallback_config)
    fallback_ms = (time.perf_counter() - fallback_started) * 1000
    common = {
        "arm": "rule",
        "aws_account_id": expected_account_id(),
        "aws_region": "us-west-2",
        "docker_version": "28.5.1",
        "environment_verification": "Verified before launch; wrapper revalidates AWS account",
        "policy_freeze_sha256": digest(freeze_path),
        "policy_mode": "enforced",
        "threshold": 0.0,
        "fallback_enabled": True,
        "fallback_config_sha256": digest(args.fallback_config),
        "task_ids": order,
        "max_tool_calls": 16,
        "decision_timeout": 30,
        "inference_environment": inference,
        "cohort_manifest_sha256": cohort_hash,
        "dataset_sha256": cohort_hash,
        "wheel_sha256": wheel_hash,
        "source_sha256": source_hashes,
        "runner_sha256": source_hashes["history_evaluate.py"],
        "device": "host-cpu",
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "task_content_sha256": task_hashes,
        "task_scenario_sha256": scenario_hashes,
        "task_scenario_raw_sha256": scenario_raw_hashes,
        "attempts_per_task": 1,
        "pairing": "alternate off/on order by interleaved task index",
        "initialization_scope": "Shared model initialized once; timing in combined manifest",
        "invoice_authorization": invoice_authorization,
        "experiment_variant": authorization.VARIANT if authorization else "history-observation-v1",
    }
    manifests = {}
    records: dict[str, list[dict[str, Any]]] = {"off": [], "on": []}
    for arm, mode in (("off", "hidden"), ("on", "observed")):
        (args.output / arm).mkdir()
        manifests[arm] = {
            **common,
            "history_mode": mode,
            "harness_variant": authorization.VARIANT
            if authorization
            else history.VARIANT
            if arm == "on"
            else "constrained-hybrid-policy-v1",
            "serialization_version": authorization.SERIALIZATION_VERSION
            if authorization
            else history.SERIALIZATION_VERSION
            if arm == "on"
            else data.SERIALIZATION_VERSION,
            "run_order": [],
        }
        write(args.output / arm / "manifest.json", manifests[arm])
    combined = {
        "schema_version": "paired-invoice-authorization-history-execution-v1"
        if invoice_authorization
        else "paired-history-execution-v1",
        "invoice_authorization": invoice_authorization,
        "experiment_variant": common["experiment_variant"],
        "tasks": len(tasks),
        "planned_trials": len(tasks) * 2,
        "cohort_manifest_sha256": cohort_hash,
        "source_sha256": source_hashes,
        "policy_freeze_sha256": digest(freeze_path),
        "aws_account_id": expected_account_id(),
        "aws_region": "us-west-2",
        "docker_version": "28.5.1",
        "fallback_load_ms_shared": fallback_ms,
        "max_job_seconds": args.max_job_seconds,
        "execution_order": [],
        "complete": False,
    }
    write(args.output / "evaluation-manifest.json", combined)
    overall_error = None
    try:
        for index, task in enumerate(order):
            for arm in ("off", "on") if index % 2 == 0 else ("on", "off"):
                if digest(args.cohort) != cohort_hash or digest(args.wheel) != wheel_hash:
                    raise RuntimeError("Frozen cohort or wheel changed during evaluation")
                if any(digest(path) != source_hashes[name] for name, path in source_paths.items()):
                    raise RuntimeError("Frozen runtime source changed during evaluation")
                if task_digest(task_paths[task]) != task_hashes[task]:
                    raise RuntimeError(f"Task contents changed during evaluation: {task}")
                remaining = args.max_job_seconds - (time.perf_counter() - started)
                if remaining < 35:
                    raise TimeoutError(
                        "Insufficient remaining job budget for another isolated trial"
                    )
                scenario = load_scenario(task_paths[task] / "scenario.yaml")
                if scenario.benchmark is None:
                    raise ValueError("Task lacks harness-resolved targets")
                position = len(combined["execution_order"])
                position_record = {"task_id": task, "arm": arm, "position": position}
                combined["execution_order"].append(position_record)
                manifests[arm]["run_order"].append(position_record)
                write(args.output / arm / "manifest.json", manifests[arm])
                write(args.output / "evaluation-manifest.json", combined)
                observed = arm == "on"
                env = NativeHarbor(
                    task_paths[task],
                    args.wheel,
                    args.output / arm / task,
                    "enforced",
                    observe_escalations=observed,
                )
                trial_started = time.perf_counter()
                result = {
                    "task_id": task,
                    "history_mode": "observed" if observed else "hidden",
                    "harness_variant": manifests[arm]["harness_variant"],
                    "execution_position": position,
                    "invoice_authorization": invoice_authorization,
                }

                async def trial(
                    env: Any = env,
                    trial_started: float = trial_started,
                    observed: bool = observed,
                    scenario: Any = scenario,
                    result: dict[str, Any] = result,
                ) -> None:
                    await env.start()
                    reset_ms = (time.perf_counter() - trial_started) * 1000
                    policy = history.rule_action if observed else data.rule_action
                    targets = scenario.benchmark.trajectory.required_inspections
                    if authorization:
                        policy = partial(
                            authorization.scoped_rule_action,
                            targets=targets,
                            escalation_history=observed,
                        )
                    episode = await run_episode(
                        request=env.task.instruction + PROMPT_SUFFIX,
                        targets=targets,
                        invoke=env.invoke,
                        predict=lambda state: {"action": policy(state), "confidence": 1.0},
                        threshold=0,
                        timeout=30,
                        fallback=fallback,
                        max_tool_calls=16,
                        escalation_history=observed,
                        invoice_authorization=invoice_authorization,
                    )
                    result.update(episode)
                    result["observed_policy_forbidden_writes"] = result[
                        "successful_forbidden_writes"
                    ]
                    result.update(
                        execution_safety(
                            result,
                            scenario.benchmark.mutation_policy.allowed_targets,
                        )
                    )
                    result["reset_ms"] = reset_ms
                    result["verifier"] = await env.verify()
                    result["verifier_pass"] = result["verifier"]["rewards"].get("reward") == 1.0

                try:
                    await asyncio.wait_for(trial(), timeout=min(300, remaining - 30))
                except Exception as exc:
                    result.update(error=f"{type(exc).__name__}: {exc}", verifier_pass=False)
                finally:
                    result["wall_ms"] = (time.perf_counter() - trial_started) * 1000
                    cleanup_started = time.perf_counter()
                    try:
                        await asyncio.wait_for(env.close(), timeout=30)
                        result["cleanup"] = "deleted-owned-environment"
                    except Exception as exc:
                        result["cleanup"] = f"failed: {exc}"
                    result["cleanup_ms"] = (time.perf_counter() - cleanup_started) * 1000
                    result["total_wall_including_cleanup_ms"] = (
                        time.perf_counter() - trial_started
                    ) * 1000
                    records[arm].append(result)
                    with (args.output / arm / "results.jsonl").open("a") as handle:
                        handle.write(json.dumps(result, sort_keys=True) + "\n")
                print(
                    json.dumps(
                        {
                            "task_id": task,
                            "arm": arm,
                            "position": position,
                            "verifier_pass": result["verifier_pass"],
                            "llm_calls": result.get("llm_calls"),
                            "error": result.get("error"),
                        }
                    ),
                    flush=True,
                )
                if result["cleanup"] != "deleted-owned-environment":
                    raise RuntimeError(
                        "Owned environment cleanup failed; stop before creating more"
                    )
    except Exception as exc:
        overall_error = f"{type(exc).__name__}: {exc}"
    finally:
        for arm in records:
            manifests[arm]["usage"] = {
                "tasks": len(records[arm]),
                "llm_calls": sum(row.get("llm_calls", 0) for row in records[arm]),
                "input_tokens": sum(row.get("llm_input_tokens", 0) for row in records[arm]),
                "output_tokens": sum(row.get("llm_output_tokens", 0) for row in records[arm]),
                "tokens_basis": "Actual provider usage; errors may omit usage",
            }
            manifests[arm]["complete"] = len(records[arm]) == len(tasks)
            write(args.output / arm / "manifest.json", manifests[arm])
        combined.update(
            complete=overall_error is None
            and all(len(rows) == len(tasks) for rows in records.values()),
            error=overall_error,
            run_total_wall_ms=(time.perf_counter() - started) * 1000,
            usage={
                key: sum(manifests[arm]["usage"][key] for arm in records)
                for key in ("llm_calls", "input_tokens", "output_tokens")
            },
        )
        write(args.output / "evaluation-manifest.json", combined)
    if overall_error:
        raise RuntimeError(overall_error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--fallback-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy-freeze", type=Path)
    parser.add_argument("--invoice-authorization", action="store_true")
    parser.add_argument("--max-job-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not 60 <= args.max_job_seconds <= 1800:
        parser.error("job budget must be between 60 and 1800 seconds")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
