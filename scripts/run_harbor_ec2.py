"""Run purpose-specific, fail-closed Harbor regressions on ephemeral EC2."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.harbor.provenance import build_clean_wheel
from agentops_demo.scale.contracts import (
    BenchmarkIdentity,
    CandidateGate,
    CandidateIdentity,
    CleanupSummary,
    Ec2Trial,
    RunPurpose,
)
from agentops_demo.scale.gate import candidate_gate
from agentops_demo.taskify.integrity import (
    harbor_task_sha256,
    manifest_scenario_sha256,
)
from agentops_demo.taskify.reproduction import ReproductionReport
from agentops_demo.validation.scenario import load_scenario
from scripts.assert_harbor_result import (
    read_trial_results,
    validate_runtime_provenance,
    with_expectations,
)
from scripts.generate_harbor_configs import write_configs
from scripts.generate_harbor_ec2_config import execution_config, job_config, load_terraform_outputs
from scripts.harbor_ec2 import DEFAULT_ROOT, EXPECTED_REGION
from scripts.harbor_ec2_controller import discover_workers, require_account

HARBOR_AGENT_MODEL = {"known-good": "scripted/correct", "scripted-bad": "scripted/bad"}
STATIC_TASK = Path("benchmarks/disputed-refund")
_PURPOSES: dict[frozenset[str], RunPurpose] = {
    frozenset({"oracle"}): "oracle_smoke",
    frozenset({"bad"}): "calibration_smoke",
    frozenset({"correct"}): "candidate_gate",
    frozenset({"scale"}): "candidate_gate",
    frozenset({"oracle", "correct", "bad"}): "parity",
}


class Ec2RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunOutcome:
    report_path: Path
    gate: CandidateGate


def purpose_for_kinds(kinds: list[str]) -> RunPurpose:
    """Validate exactly one supported run-purpose spelling before external work."""

    if not kinds or len(kinds) != len(set(kinds)):
        raise Ec2RunError("EC2 kinds must be non-empty and contain no duplicates")
    purpose = _PURPOSES.get(frozenset(kinds))
    if purpose is None:
        raise Ec2RunError("unsupported EC2 kind combination")
    return purpose


def resolve_task(
    task_source: str, scenario_dir: Path | None
) -> tuple[Path, ReproductionReport | None, BenchmarkIdentity]:
    """Bind the selected task to either Phase 5 evidence or the static task tree."""

    if task_source == "static":
        if not STATIC_TASK.is_dir():
            raise Ec2RunError(f"static Harbor task is missing: {STATIC_TASK}")
        task = STATIC_TASK.resolve()
        return (
            task,
            None,
            BenchmarkIdentity(task_source="static", harbor_task_sha256=harbor_task_sha256(task)),
        )
    if scenario_dir is None:
        raise Ec2RunError("taskify source requires --scenario-dir")
    report_path = scenario_dir / "reproduction.json"
    manifest_path = scenario_dir / "manifest.json"
    scenario_path = scenario_dir / "scenario.yaml"
    task = scenario_dir / "harbor"
    try:
        report = ReproductionReport.model_validate_json(report_path.read_text())
        scenario = load_scenario(scenario_path)
        scenario_digest = manifest_scenario_sha256(manifest_path, scenario)
        task_digest = harbor_task_sha256(task)
    except (OSError, ValueError) as exc:
        raise Ec2RunError(f"taskify artifact integrity validation failed: {exc}") from exc
    if report.status != "VALIDATED" or report.harbor_task is None:
        raise Ec2RunError("taskify source requires validated Phase 5 Harbor task evidence")
    if report.harbor_task.sha256 != task_digest:
        raise Ec2RunError("taskify Harbor task SHA-256 does not match validated evidence")
    return (
        task.resolve(),
        report,
        BenchmarkIdentity(
            task_source="taskify",
            harbor_task_sha256=task_digest,
            scenario_id=scenario.id,
            scenario_sha256=scenario_digest,
            phase5_status=report.status,
            phase5_replay_mode=report.replay_mode,
            trace_id=report.source.trace_id,
            session_id=report.source.session_id,
        ),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_job(config_path: Path) -> int:
    return subprocess.run(
        (
            "uv",
            "run",
            "--group",
            "harbor",
            "python",
            "-m",
            "scripts.run_harbor_ec2_job",
            "--config",
            str(config_path),
        ),
        check=False,
    ).returncode


def _worker_ids(job_dir: Path) -> list[str]:
    """Best-effort worker IDs retained by Harbor diagnostics, never launch counts."""

    found: set[str] = set()
    for path in job_dir.rglob("*.json"):
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        stack: list[object] = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, child in item.items():
                    if key in {"instance_id", "InstanceId"} and isinstance(child, str):
                        found.add(child)
                    else:
                        stack.append(child)
            elif isinstance(item, list):
                stack.extend(item)
    with suppress(OSError):
        found.update(re.findall(r"\bi-[0-9a-f]+\b", (job_dir / "job.log").read_text()))
    return sorted(found)


def _cleanup_outcome(run_id: str) -> tuple[list[str], list[str], str | None]:
    """Observe tagged workers after jobs; explicit cleanup remains separate authority."""

    import boto3

    try:
        session = boto3.Session(
            profile_name=os.getenv("AWS_PROFILE", "default"), region_name=EXPECTED_REGION
        )
        require_account(session.client("sts"))
        remaining = discover_workers(session.client("ec2"), run_id)
    except Exception as exc:  # Report the observation failure rather than losing evidence.
        return [], [], str(exc)
    identifiers = sorted(
        instance["InstanceId"]
        for instance in remaining
        if isinstance(instance.get("InstanceId"), str)
    )
    return [], identifiers, None


def _retained_trials(
    *,
    name: str,
    job_dir: Path,
    expected_reward: float,
    config: AgentConfig | None,
    package_sha256: str | None,
    model: str,
) -> list[Ec2Trial]:
    """Preserve parsed attempts; represent unreadable evidence as non-retained."""

    try:
        trial_name = (
            "oracle"
            if name == "oracle"
            else "known-good"
            if name == "known-good"
            else "originating"
        )
        retained: list[Ec2Trial] = []
        for observed in read_trial_results(trial_name, job_dir):
            observed = with_expectations(observed, expected_reward)
            provenance_matches = True
            infrastructure_failure: str | None = None
            if config is not None and package_sha256 is not None:
                try:
                    validate_runtime_provenance(
                        observed, config=config, package_sha256=package_sha256, model=model
                    )
                except AssertionError:
                    provenance_matches = False
            retained.append(
                Ec2Trial(
                    name=name,
                    requested_reward=expected_reward,
                    retained=True,
                    reward=observed.reward,
                    completed=observed.reward is not None,
                    provenance_matches=provenance_matches,
                    infrastructure_failure=infrastructure_failure,
                    tool_calls=observed.tool_calls,
                    config_fingerprint=observed.config_fingerprint,
                    package_sha256=observed.package_sha256,
                    model=observed.model,
                )
            )
        if not retained:
            return [
                Ec2Trial(
                    name=name,
                    requested_reward=expected_reward,
                    retained=False,
                    completed=False,
                    provenance_matches=False,
                    infrastructure_failure="Harbor retained no structured trial results",
                )
            ]
        return retained
    except (AssertionError, KeyError, ValueError) as exc:
        return [
            Ec2Trial(
                name=name,
                requested_reward=expected_reward,
                retained=False,
                completed=False,
                provenance_matches=False,
                infrastructure_failure=str(exc),
            )
        ]


def _job_specs(
    purpose: RunPurpose, configs: dict[str, AgentConfig]
) -> list[tuple[str, str, float, AgentConfig | None]]:
    specs: list[tuple[str, str, float, AgentConfig | None]] = []
    if purpose in {"oracle_smoke", "parity"}:
        specs.append(("oracle", "oracle", 1.0, None))
    if purpose in {"candidate_gate", "parity"}:
        specs.append(("known-good", HARBOR_AGENT_MODEL["known-good"], 1.0, configs["correct"]))
    if purpose in {"calibration_smoke", "parity"}:
        specs.append(("scripted-bad", HARBOR_AGENT_MODEL["scripted-bad"], 0.0, configs["bad"]))
    return specs


def _validate_execution_shape(
    purpose: RunPurpose, execution_attempts: int, concurrency: int
) -> None:
    if purpose == "parity" and (execution_attempts, concurrency) != (1, 1):
        raise Ec2RunError("parity requires exactly one serial attempt per control")


def run(
    *,
    task_source: str,
    scenario_dir: Path | None,
    outputs: Path,
    run_id: str,
    kinds: list[str],
) -> RunOutcome:
    """Execute one purpose and always persist its meaningful acceptance report."""

    purpose = purpose_for_kinds(kinds)
    if purpose == "parity" and task_source != "taskify":
        raise Ec2RunError("parity requires a validated taskify regression")
    task, source_report, benchmark = resolve_task(task_source, scenario_dir)
    execution = execution_config(
        outputs=load_terraform_outputs(outputs), run_id=run_id, task_source=task_source
    )
    if kinds == ["scale"] and (execution.attempts, execution.concurrency) != (4, 4):
        raise Ec2RunError("scale requires HARBOR_EC2_ATTEMPTS=4 and HARBOR_EC2_CONCURRENCY=4")
    _validate_execution_shape(purpose, execution.attempts, execution.concurrency)
    root = DEFAULT_ROOT / "runs" / run_id
    jobs_root = root / "jobs"
    configs_root = root / "configs"
    _write_json(root / "execution.json", execution.model_dump(mode="json"))
    wheel, package = build_clean_wheel(root / "package")
    configs = write_configs(configs_root, package.source_revision)
    private_key = Path(os.environ["HARBOR_EC2_SSH_PRIVATE_KEY"])
    trials: list[Ec2Trial] = []
    observed_workers: set[str] = set()
    for name, model, expected, config in _job_specs(purpose, configs):
        job_dir = jobs_root / name
        config_path = root / "job-configs" / f"{name}.json"
        payload = job_config(
            execution=execution,
            task=task,
            jobs_dir=jobs_root,
            job_name=name,
            model=model,
            agent_config_path=(
                configs_root / ("correct.json" if name == "known-good" else "bad.json")
                if config
                else None
            ),
            package_path=wheel if config else None,
            ssh_key_path=private_key,
        )
        _write_json(config_path, payload)
        started = time.monotonic()
        returncode = _run_job(config_path)
        retained = _retained_trials(
            name=name,
            job_dir=job_dir,
            expected_reward=expected,
            config=config,
            package_sha256=package.sha256 if config else None,
            model=model,
        )
        workers = _worker_ids(job_dir)
        observed_workers.update(workers)
        for trial in retained:
            update: dict[str, object] = {
                "elapsed_seconds": time.monotonic() - started,
                "worker_instance_ids": workers,
            }
            if returncode and trial.infrastructure_failure is None:
                update.update(infrastructure_failure=f"Harbor exited {returncode}")
            trials.append(Ec2Trial.model_validate({**trial.model_dump(), **update}))
    _, remaining_workers, cleanup_error = _cleanup_outcome(run_id)
    cleanup = CleanupSummary(
        workers_observed=sorted(observed_workers),
        workers_remaining=remaining_workers,
        passed=not remaining_workers and cleanup_error is None,
        error=cleanup_error,
    )
    if remaining_workers:
        for index, trial in enumerate(trials):
            trials[index] = Ec2Trial.model_validate(
                {
                    **trial.model_dump(),
                    "worker_instance_ids": sorted(
                        set(trial.worker_instance_ids + remaining_workers)
                    ),
                }
            )
    parity: bool | None = None
    if purpose == "parity":
        assert source_report is not None
        source_trials = {trial.name: trial.reward for trial in source_report.trials}
        expected_local = {
            "oracle": source_trials.get("oracle"),
            "known-good": source_trials.get("known-good"),
            "scripted-bad": source_trials.get("originating"),
        }
        parity = all(
            trial.completed and trial.reward == expected_local[trial.name] for trial in trials
        )
    candidate = None
    if purpose == "candidate_gate":
        candidate = CandidateIdentity(
            name="known-good",
            model="scripted/correct",
            agent_config_fingerprint=configs["correct"].fingerprint(),
            package_sha256=package.sha256,
        )
    gate = candidate_gate(
        run_id=run_id,
        purpose=purpose,
        task_source=task_source,
        execution=execution,
        benchmark=benchmark,
        execution_package=package,
        candidate=candidate,
        trials=trials,
        cleanup=cleanup,
        local_ec2_parity=parity,
    )
    report_path = root / "candidate-gate.json"
    _write_json(report_path, gate.model_dump(mode="json"))
    return RunOutcome(report_path=report_path, gate=gate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_ROOT / "terraform-outputs.json")
    parser.add_argument("--scenario-dir", type=Path)
    parser.add_argument(
        "--task-source",
        choices=("taskify", "static"),
        default=os.getenv("HARBOR_TASK_SOURCE", "taskify"),
    )
    parser.add_argument(
        "--run-id", default=os.getenv("HARBOR_EC2_RUN_ID", f"run-{uuid.uuid4().hex[:12]}")
    )
    parser.add_argument(
        "--kind", action="append", choices=("oracle", "correct", "bad", "scale"), required=True
    )
    args = parser.parse_args(argv)
    outcome = run(
        task_source=args.task_source,
        scenario_dir=args.scenario_dir,
        outputs=args.outputs,
        run_id=args.run_id,
        kinds=args.kind,
    )
    print(outcome.report_path)
    return 0 if outcome.gate.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
