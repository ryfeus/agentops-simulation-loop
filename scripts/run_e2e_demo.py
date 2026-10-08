"""Guarded Phase 7 orchestration for one linked AgentOps evidence run."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from agentops_demo.agentcore.candidates import SUPPORTED_CANDIDATES, resolve_candidate
from agentops_demo.agentcore.request import CandidateSelector
from agentops_demo.demo.contracts import DemoReport, ProductionFailureIdentity, StageResult
from agentops_demo.demo.lineage import validate_lineage
from agentops_demo.demo.verification import verify_demo_bundle
from agentops_demo.harbor.provenance import current_revision, require_clean_worktree
from agentops_demo.scale.contracts import CandidateGate
from agentops_demo.taskify.contracts import FailureEvidence
from agentops_demo.taskify.harbor_runner import reproduce
from agentops_demo.taskify.reproduction import ReproductionReport
from agentops_demo.taskify.trace import trace_sha256
from agentops_demo.validation.scenario import load_scenario
from scripts import run_harbor_ec2, taskify
from scripts.aws_context import expected_account_id

ROOT = Path(".demo/runs")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _stage(
    root: Path,
    number: int,
    stage: str,
    started: str,
    artifacts: dict[str, Path],
    reason: str | None = None,
) -> None:
    result = StageResult(
        stage=stage,
        status="PASS" if reason is None else "FAIL",
        started_at=started,
        finished_at=_now(),
        artifacts={key: _sha(path) for key, path in artifacts.items()},
        artifact_paths={key: str(path.relative_to(root)) for key, path in artifacts.items()},
        reason=reason,
    )
    _write(
        root / "stages" / f"{number:02d}-{stage.replace('_', '-')}.json",
        result.model_dump(mode="json"),
    )


def _completed(root: Path, number: int, stage: str) -> bool:
    """Validate a completed stage before a resume can reuse any evidence."""

    path = root / "stages" / f"{number:02d}-{stage.replace('_', '-')}.json"
    if not path.is_file():
        return False
    result = StageResult.model_validate_json(path.read_text())
    if result.status != "PASS":
        return False
    for name, digest in result.artifacts.items():
        artifact = root / result.artifact_paths[name]
        if not artifact.is_file() or _sha(artifact) != digest:
            raise RuntimeError(f"cannot resume: {stage} artifact {name} is missing or altered")
    return True


def _validate_resume(root: Path) -> dict[str, object]:
    try:
        run = json.loads((root / "run.json").read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot resume without a valid run.json") from exc
    if run.get("git_revision") != current_revision():
        raise RuntimeError("cannot resume from a different Git revision")
    stages = (
        "preflight",
        "production_failure",
        "taskify",
        "local_regression",
        "ec2_parity",
        "ec2_scale",
    )
    for number, stage in enumerate(stages, start=1):
        _completed(root, number, stage)
    return run


def preflight(run_id: str, *, candidate: CandidateSelector | None = "bedrock") -> None:
    """Read-only cloud preflight; it deliberately performs no deployment or trial."""
    require_clean_worktree()

    from scripts.agentcore_package import load_config
    from scripts.aws_context import verified_session
    from scripts.dsql_admin import read_state
    from scripts.evaluation_status import main as evaluation_status
    from scripts.generate_harbor_ec2_config import load_terraform_outputs
    from scripts.harbor_ec2_preflight import preflight as ec2_preflight
    from scripts.observability_status import main as observability_status

    session = verified_session()
    if candidate is not None:
        resolve_candidate(load_config(), candidate)
    outputs = load_terraform_outputs(Path(".harbor-ec2/terraform-outputs.json"))
    ec2_preflight(
        sts_client=session.client("sts"), ec2_client=session.client("ec2"), outputs=outputs
    )
    if not asyncio.run(read_state()):
        raise RuntimeError("DSQL health check returned an empty state")
    if observability_status() != 0 or evaluation_status() != 0:
        raise RuntimeError("AgentCore observability or online evaluation health check failed")
    if run_harbor_ec2.discover_workers(session.client("ec2"), run_id):
        raise RuntimeError("tagged Phase 6 workers already exist for this demo run")


def acquire_live(root: Path, *, candidate: CandidateSelector) -> tuple[Path, Path]:
    started = _now()
    subprocess.run(
        (
            "uv",
            "run",
            "python",
            "-m",
            "scripts.trigger_online_failure",
            "--candidate",
            candidate,
        ),
        check=True,
    )
    failure = Path(".agentcore/evaluation/failure.json")
    destination = root / "production"
    destination.mkdir(parents=True, exist_ok=True)
    frozen_failure = destination / "failure.json"
    shutil.copyfile(failure, frozen_failure)
    evidence = FailureEvidence.model_validate_json(frozen_failure.read_text())
    trace = taskify.fetch_exact_trace(evidence)
    frozen_trace = destination / "trace.json"
    _write(frozen_trace, trace)
    _stage(
        root, 2, "production_failure", started, {"failure": frozen_failure, "trace": frozen_trace}
    )
    return frozen_failure, frozen_trace


def acquire_replay(root: Path, source_run: Path) -> tuple[Path, Path]:
    started = _now()
    report = verify_demo_bundle(source_run)
    if report.mode != "live" or report.source.source != "LIVE":
        raise RuntimeError("replay source must be an independently verified live demo")
    source_failure = source_run / "production/failure.json"
    source_trace = source_run / "production/trace.json"
    if _sha(source_failure) != report.source.failure_sha256:
        raise RuntimeError("replay source failure digest does not match its accepted report")
    if trace_sha256(json.loads(source_trace.read_text())) != report.source.trace_sha256:
        raise RuntimeError("replay source trace digest does not match its accepted report")
    destination = root / "production"
    destination.mkdir(parents=True, exist_ok=True)
    failure, trace = destination / "failure.json", destination / "trace.json"
    shutil.copyfile(source_failure, failure)
    shutil.copyfile(source_trace, trace)
    _stage(root, 2, "production_failure", started, {"failure": failure, "trace": trace})
    return failure, trace


def taskify_stage(root: Path, failure: Path, trace: Path) -> Path:
    started = _now()
    scenario_dir = taskify.create(
        failure_path=failure, trace_path=trace, output_root=root / "taskify"
    )
    scenario = scenario_dir / "scenario.yaml"
    load_scenario(scenario)
    manifest = json.loads((scenario_dir / "manifest.json").read_text())
    evidence = FailureEvidence.model_validate_json(failure.read_text())
    if (
        manifest.get("source", {}).get("failure_sha256") != _sha(failure)
        or manifest.get("source", {}).get("trace_sha256")
        != trace_sha256(json.loads(trace.read_text()))
        or manifest.get("source", {}).get("trace_id") != evidence.source.trace_id
        or manifest.get("source", {}).get("session_id") != evidence.source.session_id
    ):
        raise RuntimeError("taskify manifest does not bind the frozen production evidence")
    _stage(
        root,
        3,
        "taskify",
        started,
        {"scenario": scenario, "manifest": scenario_dir / "manifest.json"},
    )
    return scenario_dir


def local_stage(root: Path, scenario_dir: Path) -> ReproductionReport:
    started = _now()
    report = reproduce(scenario_dir / "scenario.yaml")
    if report.status != "VALIDATED":
        raise RuntimeError(f"local regression did not validate: {report.status}")
    path = scenario_dir / "reproduction.json"
    _stage(root, 4, "local_regression", started, {"reproduction": path})
    return report


def ec2_stage(
    root: Path, run_id: str, scenario_dir: Path, kind: list[str], name: str
) -> CandidateGate:
    started = _now()
    outcome = run_harbor_ec2.run(
        task_source="taskify",
        scenario_dir=scenario_dir,
        outputs=Path(".harbor-ec2/terraform-outputs.json"),
        run_id=f"{run_id}-{name}",
        kinds=kind,
    )
    if not outcome.gate.accepted:
        raise RuntimeError("EC2 gate was not accepted")
    target = root / "ec2" / name / "candidate-gate.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(outcome.report_path, target)
    _stage(root, 5 if name == "parity" else 6, f"ec2_{name}", started, {"gate": target})
    return outcome.gate


def run(
    *,
    run_id: str,
    mode: str,
    candidate: CandidateSelector | None = None,
    source_run: Path | None = None,
    resume: bool = False,
) -> Path:
    if mode == "replay" and source_run is None and not resume:
        raise RuntimeError(
            "replay mode requires --source-run; live mode never falls back to replay"
        )
    root = ROOT / run_id
    if root.exists() and not resume:
        raise RuntimeError(f"demo run already exists: {root}")
    if resume:
        run_metadata = _validate_resume(root)
        if run_metadata.get("mode") != mode:
            raise RuntimeError("cannot resume with a different demo mode")
        stored_candidate = run_metadata.get("candidate")
        if mode == "live":
            if stored_candidate not in SUPPORTED_CANDIDATES:
                raise RuntimeError("cannot resume live run without a valid candidate")
            if candidate is not None and candidate != stored_candidate:
                raise RuntimeError("cannot resume with a different candidate")
            candidate = stored_candidate
        elif candidate is not None:
            raise RuntimeError("replay mode does not accept a candidate override")
        if mode == "replay" and source_run is None:
            stored_source = run_metadata.get("source_run")
            if not isinstance(stored_source, str) or not stored_source:
                raise RuntimeError("cannot resume replay without the original source run")
            source_run = Path(stored_source)
        started = str(run_metadata["started_at"])
    else:
        if mode == "live":
            candidate = candidate or "bedrock"
        elif candidate is not None:
            raise RuntimeError("replay mode does not accept a candidate override")
        started = _now()
        root.mkdir(parents=True)
        _write(
            root / "run.json",
            {
                "schema_version": "1",
                "run_id": run_id,
                "mode": mode,
                "started_at": started,
                "git_revision": current_revision(),
                "aws_account": expected_account_id(),
                "aws_region": "us-west-2",
                "candidate": candidate,
                "source_run": str(source_run) if source_run else None,
            },
        )
    if not _completed(root, 1, "preflight"):
        preflight_started = _now()
        preflight(run_id, candidate=candidate)
        _stage(root, 1, "preflight", preflight_started, {"run": root / "run.json"})
    if _completed(root, 2, "production_failure"):
        failure, trace = root / "production/failure.json", root / "production/trace.json"
    else:
        if mode == "live":
            assert candidate is not None
            failure, trace = acquire_live(root, candidate=candidate)
        else:
            failure, trace = acquire_replay(root, source_run)
    evidence = FailureEvidence.model_validate_json(failure.read_text())
    if _completed(root, 3, "taskify"):
        scenario_dir = next((root / "taskify").glob("*/scenario.yaml")).parent
    else:
        scenario_dir = taskify_stage(root, failure, trace)
    if _completed(root, 4, "local_regression"):
        local = ReproductionReport.model_validate_json(
            (scenario_dir / "reproduction.json").read_text()
        )
    else:
        local = local_stage(root, scenario_dir)
    if _completed(root, 5, "ec2_parity"):
        parity = CandidateGate.model_validate_json(
            (root / "ec2/parity/candidate-gate.json").read_text()
        )
    else:
        parity = ec2_stage(root, run_id, scenario_dir, ["oracle", "correct", "bad"], "parity")
    if _completed(root, 6, "ec2_scale"):
        scale = CandidateGate.model_validate_json(
            (root / "ec2/scale/candidate-gate.json").read_text()
        )
    else:
        scale = ec2_stage(root, run_id, scenario_dir, ["scale"], "scale")
    source = ProductionFailureIdentity(
        source="LIVE" if mode == "live" else "REPLAY",
        model=f"{evidence.candidate.model_provider}/{evidence.candidate.model_id}",
        session_id=evidence.source.session_id,
        trace_id=evidence.source.trace_id,
        source_revision=evidence.candidate.source_revision,
        agent_config_fingerprint=evidence.candidate.agent_config_fingerprint,
        failure_sha256=_sha(failure),
        trace_sha256=trace_sha256(json.loads(trace.read_text())),
    )
    lineage = validate_lineage(source, str(scenario_dir / "scenario.yaml"), local, parity, scale)
    cleanup = scale.cleanup
    accepted = lineage.passed and parity.accepted and scale.accepted and scale.eligible is True
    report = DemoReport(
        run_id=run_id,
        mode=mode,
        started_at=started,
        finished_at=_now(),
        git_revision=current_revision(),
        source=source,
        calibration=local.calibration,
        local_validation=local,
        ec2_parity=parity,
        ec2_scale=scale,
        lineage=lineage,
        cleanup=cleanup,
        accepted=accepted,
        failure_reasons=[] if accepted else ["one or more gates failed"],
    )
    report_path = root / "demo-report.json"
    _write(report_path, report.model_dump(mode="json"))
    summary = root / "demo-summary.md"
    summary.write_text(
        f"# AgentOps Loop Demo — {'PASS' if accepted else 'FAIL'}\n\nMode: {mode.upper()}\n\n"
        f"Scenario: `{scenario_dir.name}`\n\nOracle / known-good / calibration: `1 / 1 / 0`\n\n"
        f"Candidate trials: `4 / 4`; eligible: `{scale.eligible}`\n\n"
        f"Workers remaining: `{len(cleanup.workers_remaining)}`\n"
    )
    _stage(root, 7, "finalize", _now(), {"report": report_path, "summary": summary})
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"demo-{datetime.now(UTC):%Y%m%d%H%M%S}")
    parser.add_argument("--mode", choices=("live", "replay"))
    parser.add_argument("--candidate", choices=SUPPORTED_CANDIDATES)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    mode = args.mode
    if args.resume and mode is None:
        try:
            mode = json.loads((ROOT / args.run_id / "run.json").read_text())["mode"]
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"cannot infer resume mode: {exc}")
    root = run(
        run_id=args.run_id,
        mode=mode or "live",
        candidate=args.candidate,
        source_run=args.source_run,
        resume=args.resume,
    )
    report = verify_demo_bundle(root)
    print(root / "demo-summary.md")
    return 0 if report.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
