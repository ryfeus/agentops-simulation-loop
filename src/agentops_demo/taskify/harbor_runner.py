"""Execute generated Harbor tasks and derive trustworthy reproduction evidence."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.harbor.provenance import PackageProvenance, build_clean_wheel
from agentops_demo.taskify.harbor_renderer import render_harbor_task
from agentops_demo.taskify.integrity import harbor_task_sha256
from agentops_demo.taskify.reproduction import (
    CalibrationIdentity,
    HarborTaskIdentity,
    ReproductionReport,
    SourceProvenance,
    TrialResult,
)
from agentops_demo.validation.scenario import load_scenario
from scripts.assert_harbor_result import (
    read_trial_result,
    validate_runtime_provenance,
    validate_trial,
    with_expectations,
)

HARBOR_AGENT = "agentops_demo.harbor.agent:LangGraphBillingAgent"


class HarborReplayError(RuntimeError):
    pass


def _trace_parts(trace_ref: str) -> tuple[str, str]:
    try:
        before_trace, trace_id = trace_ref.rsplit("/trace/", 1)
        session_id = before_trace.rsplit("/session/", 1)[1]
    except IndexError as exc:
        raise HarborReplayError("scenario has an invalid trace provenance reference") from exc
    return session_id, trace_id


def _write_config(path: Path, config: AgentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")


def _run_harbor(
    *,
    task: Path,
    jobs_root: Path,
    name: str,
    model: str,
    config: Path | None = None,
    wheel: Path | None = None,
) -> Path:
    command = [
        "uv",
        "run",
        "--group",
        "harbor",
        "harbor",
        "run",
        "-p",
        str(task),
        "-a",
        "oracle" if config is None else HARBOR_AGENT,
        "-m",
        model,
        "--env",
        "docker",
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--yes",
        "--job-name",
        name,
        "--jobs-dir",
        str(jobs_root),
    ]
    if config is not None and wheel is not None:
        command.extend(("--ak", f"agent_config_path={config.resolve()}"))
        command.extend(("--ak", f"package_path={wheel}"))
    completed = subprocess.run(command, check=False)
    job_dir = jobs_root / name
    if completed.returncode and not (job_dir / "result.json").is_file():
        raise HarborReplayError(f"Harbor {name} did not write a structured result")
    return job_dir


def _empty_trial(
    name: str, job_dir: Path, expected_reward: float, expected_tools: list[str] | None
) -> TrialResult:
    return with_expectations(
        TrialResult(name=name, attempted=True, job_dir=str(job_dir)),
        expected_reward,
        expected_tools,
    )


def _observe_trial(
    name: str, job_dir: Path, expected_reward: float, expected_tools: list[str] | None
) -> TrialResult:
    try:
        observed = read_trial_result(name, job_dir)
    except (AssertionError, KeyError, ValueError) as exc:
        raise HarborReplayError(f"Harbor {name} result is malformed: {exc}") from exc
    return with_expectations(observed, expected_reward, expected_tools)


def _report(
    *,
    status: str,
    source_kind: str,
    source: SourceProvenance,
    calibration: CalibrationIdentity | None,
    package: PackageProvenance | None,
    trials: list[TrialResult],
    harbor_task: HarborTaskIdentity | None = None,
    reason: str | None = None,
) -> ReproductionReport:
    originating = next((trial for trial in trials if trial.name == "originating"), None)
    attempts = int(originating is not None and originating.attempted)
    failures = int(originating is not None and originating.reward == 0.0)
    return ReproductionReport(
        status=status,
        replay_mode="calibration",
        source_kind=source_kind,
        source=source,
        calibration=calibration,
        execution_package=package,
        harbor_task=harbor_task,
        attempts=attempts,
        failure_count=failures,
        trials=trials,
        reason=reason,
    )


def _write_report(root: Path, report: ReproductionReport) -> None:
    (root / "reproduction.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )


def _calibration(
    config: AgentConfig, tools: list[str]
) -> tuple[AgentConfig, CalibrationIdentity] | str:
    if config.model.provider == "scripted" and config.model.model_id == "bad":
        return config, CalibrationIdentity(
            kind="source_replay",
            model="scripted/bad",
            agent_config_fingerprint=config.fingerprint(),
            expected_tools=tools,
            derived_from_observed_failure=False,
        )
    if config.model.provider == "bedrock" and tools == ["get_invoice", "refund_invoice"]:
        raw = config.model_dump(mode="json")
        raw["model"] = {"provider": "scripted", "model_id": "bad"}
        calibrated = AgentConfig.model_validate(raw)
        return calibrated, CalibrationIdentity(
            kind="trajectory_calibration",
            model="scripted/bad",
            agent_config_fingerprint=calibrated.fingerprint(),
            expected_tools=tools,
            derived_from_observed_failure=True,
        )
    return "no-network Harbor supports only scripted/bad or the disputed-refund Bedrock trajectory"


def reproduce(scenario_path: Path) -> ReproductionReport:
    """Render and execute supported quality gates for one canonical Scenario."""

    scenario = load_scenario(scenario_path)
    root = scenario_path.parent
    source_config = scenario.provenance.originating_agent_config
    if source_config is None:
        raise HarborReplayError("scenario has no originating candidate configuration")
    session_id, trace_id = _trace_parts(scenario.provenance.source.trace_ref)
    source = SourceProvenance(
        source_revision=source_config.agent.source_revision,
        trace_id=trace_id,
        session_id=session_id,
        agent_config_fingerprint=source_config.fingerprint(),
        model=f"{source_config.model.provider}/{source_config.model.model_id}",
    )
    tools = [call.tool for call in scenario.observed_failure.tool_calls]
    selected = _calibration(source_config, tools)
    if isinstance(selected, str):
        report = _report(
            status="UNSUPPORTED",
            source_kind=scenario.provenance.source.kind,
            source=source,
            calibration=None,
            package=None,
            trials=[],
            reason=selected,
        )
        _write_report(root, report)
        return report

    calibration_config, calibration = selected

    task = root / "harbor"
    asyncio.run(render_harbor_task(scenario, task))
    harbor_task = HarborTaskIdentity(sha256=harbor_task_sha256(task))
    candidate_dir = root / "candidate"
    originating_path = candidate_dir / "originating-agent-config.json"
    _write_config(originating_path, calibration_config)
    good_raw = source_config.model_dump(mode="json")
    good_raw["model"] = {"provider": "scripted", "model_id": "correct"}
    good_config = AgentConfig.model_validate(good_raw)
    good_path = candidate_dir / "known-good-agent-config.json"
    _write_config(good_path, good_config)

    # This guard runs before any Harbor job and binds calibration to a clean checkout.
    wheel, package = build_clean_wheel(root / "package")
    jobs = root / "jobs"
    jobs.mkdir(exist_ok=True)
    trials: list[TrialResult] = []
    controls = (
        ("oracle", "oracle", None, None, 1.0, None),
        (
            "known-good",
            "scripted/correct",
            good_path,
            good_config,
            1.0,
            ["get_invoice", "escalate_dispute"],
        ),
    )
    for name, model, config_path, config, expected_reward, expected_tools in controls:
        job_dir = jobs / name
        try:
            _run_harbor(
                task=task,
                jobs_root=jobs,
                name=name,
                model=model,
                config=config_path,
                wheel=wheel if config_path is not None else None,
            )
            trial = _observe_trial(name, job_dir, expected_reward, expected_tools)
        except HarborReplayError as exc:
            trials.append(_empty_trial(name, job_dir, expected_reward, expected_tools))
            report = _report(
                status="NEEDS_REVIEW",
                source_kind=scenario.provenance.source.kind,
                source=source,
                calibration=calibration,
                package=package,
                trials=trials,
                harbor_task=harbor_task,
                reason=str(exc),
            )
            _write_report(root, report)
            return report
        trials.append(trial)
        try:
            validate_trial(trial)
            if config is not None:
                validate_runtime_provenance(
                    trial,
                    config=config,
                    package_sha256=package.sha256,
                    model=model,
                )
        except AssertionError as exc:
            report = _report(
                status="NEEDS_REVIEW",
                source_kind=scenario.provenance.source.kind,
                source=source,
                calibration=calibration,
                package=package,
                trials=trials,
                harbor_task=harbor_task,
                reason=str(exc),
            )
            _write_report(root, report)
            return report

    source_job = jobs / "originating"
    try:
        _run_harbor(
            task=task,
            jobs_root=jobs,
            name="originating",
            model="scripted/bad",
            config=originating_path,
            wheel=wheel,
        )
        originating = _observe_trial(
            "originating", source_job, 0.0, ["get_invoice", "refund_invoice"]
        )
    except HarborReplayError as exc:
        trials.append(
            _empty_trial("originating", source_job, 0.0, ["get_invoice", "refund_invoice"])
        )
        report = _report(
            status="NEEDS_REVIEW",
            source_kind=scenario.provenance.source.kind,
            source=source,
            calibration=calibration,
            package=package,
            trials=trials,
            harbor_task=harbor_task,
            reason=str(exc),
        )
        _write_report(root, report)
        return report
    trials.append(originating)
    if originating.reward is None:
        status, reason = "NEEDS_REVIEW", "originating Harbor trial has no numeric reward"
    elif originating.reward != 0.0:
        status, reason = "NOT_REPRODUCED", "originating source behavior did not fail"
    else:
        try:
            validate_trial(originating)
            validate_runtime_provenance(
                originating,
                config=calibration_config,
                package_sha256=package.sha256,
                model="scripted/bad",
            )
        except AssertionError as exc:
            status, reason = "NEEDS_REVIEW", str(exc)
        else:
            status, reason = "VALIDATED", "calibration reproduced the supported source failure"
    report = _report(
        status=status,
        source_kind=scenario.provenance.source.kind,
        source=source,
        calibration=calibration,
        package=package,
        trials=trials,
        harbor_task=harbor_task,
        reason=reason,
    )
    _write_report(root, report)
    return report
