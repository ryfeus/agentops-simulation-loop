"""Fail-closed Phase 6 purpose, evidence, and report regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentops_demo.harbor.provenance import PackageProvenance
from agentops_demo.scale.contracts import (
    BenchmarkIdentity,
    CandidateIdentity,
    CleanupSummary,
    Ec2ExecutionConfig,
    Ec2Trial,
)
from agentops_demo.scale.gate import candidate_gate
from agentops_demo.taskify.contracts import FailureEvidence
from agentops_demo.taskify.integrity import harbor_task_sha256, scenario_sha256
from agentops_demo.taskify.reproduction import (
    CalibrationIdentity,
    HarborTaskIdentity,
    ReproductionReport,
    SourceProvenance,
    TrialResult,
)
from agentops_demo.taskify.scenario_builder import build_scenario
from agentops_demo.validation.scenario import dump_scenario_yaml
from scripts import assert_phase6_acceptance, run_harbor_ec2

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/taskify"


def _execution(
    *, source: str = "static", attempts: int = 1, concurrency: int = 1
) -> Ec2ExecutionConfig:
    return Ec2ExecutionConfig(
        vpc_id="vpc-1",
        subnet_id="subnet-1",
        security_group_id="sg-1",
        key_name="key",
        ami_id="ami-1",
        run_id="run-1",
        task_source=source,  # type: ignore[arg-type]
        attempts=attempts,
        concurrency=concurrency,
        tags={"Project": "agentops-demo", "Phase": "6", "RunId": "run-1"},
    )


def _package() -> PackageProvenance:
    return PackageProvenance(
        source_revision="a" * 40,
        filename="candidate.whl",
        sha256="b" * 64,
        worktree_clean=True,
    )


def _benchmark(source: str = "static") -> BenchmarkIdentity:
    if source == "taskify":
        return BenchmarkIdentity(
            task_source="taskify",
            harbor_task_sha256="c" * 64,
            scenario_id="scenario",
            scenario_sha256="d" * 64,
            phase5_status="VALIDATED",
            phase5_replay_mode="calibration",
            trace_id="trace",
            session_id="session",
        )
    return BenchmarkIdentity(task_source="static", harbor_task_sha256="c" * 64)


def _candidate() -> CandidateIdentity:
    return CandidateIdentity(
        name="known-good",
        model="scripted/correct",
        agent_config_fingerprint="d" * 64,
        package_sha256="b" * 64,
    )


def _trial(name: str, expected: float, **changes: object) -> Ec2Trial:
    data: dict[str, object] = {
        "name": name,
        "requested_reward": expected,
        "retained": True,
        "reward": expected,
        "completed": True,
        "provenance_matches": True,
    }
    if name == "known-good":
        data.update(
            config_fingerprint="d" * 64,
            package_sha256="b" * 64,
            model="scripted/correct",
        )
    data.update(changes)
    return Ec2Trial.model_validate(data)


def _cleanup(*, passed: bool = True) -> CleanupSummary:
    return CleanupSummary(
        workers_observed=["i-observed"],
        workers_remaining=[] if passed else ["i-left"],
        passed=passed,
    )


@pytest.mark.parametrize(
    ("kinds", "purpose"),
    [
        (["oracle"], "oracle_smoke"),
        (["bad"], "calibration_smoke"),
        (["correct"], "candidate_gate"),
        (["scale"], "candidate_gate"),
        (["oracle", "correct", "bad"], "parity"),
    ],
)
def test_kind_combinations_map_to_exact_purpose(kinds: list[str], purpose: str) -> None:
    assert run_harbor_ec2.purpose_for_kinds(kinds) == purpose


@pytest.mark.parametrize(
    "kinds", [[], ["oracle", "oracle"], ["oracle", "bad"], ["scale", "correct"]]
)
def test_kind_combinations_reject_unsupported_or_duplicate_values(kinds: list[str]) -> None:
    with pytest.raises(run_harbor_ec2.Ec2RunError):
        run_harbor_ec2.purpose_for_kinds(kinds)


def test_purpose_gate_semantics_cover_smoke_parity_and_candidate() -> None:
    oracle = candidate_gate(
        run_id="run-1",
        purpose="oracle_smoke",
        task_source="static",
        execution=_execution(),
        benchmark=_benchmark(),
        execution_package=_package(),
        candidate=None,
        trials=[_trial("oracle", 1.0)],
        cleanup=_cleanup(),
        local_ec2_parity=None,
    )
    calibration = candidate_gate(
        run_id="run-1",
        purpose="calibration_smoke",
        task_source="static",
        execution=_execution(),
        benchmark=_benchmark(),
        execution_package=_package(),
        candidate=None,
        trials=[_trial("scripted-bad", 0.0)],
        cleanup=_cleanup(),
        local_ec2_parity=None,
    )
    parity_execution = _execution(source="taskify")
    parity = candidate_gate(
        run_id="run-1",
        purpose="parity",
        task_source="taskify",
        execution=parity_execution,
        benchmark=_benchmark("taskify"),
        execution_package=_package(),
        candidate=None,
        trials=[_trial("oracle", 1.0), _trial("known-good", 1.0), _trial("scripted-bad", 0.0)],
        cleanup=_cleanup(),
        local_ec2_parity=True,
    )
    candidate = candidate_gate(
        run_id="run-1",
        purpose="candidate_gate",
        task_source="static",
        execution=_execution(attempts=4, concurrency=4),
        benchmark=_benchmark(),
        execution_package=_package(),
        candidate=_candidate(),
        trials=[_trial("known-good", 1.0) for _ in range(4)],
        cleanup=_cleanup(),
        local_ec2_parity=None,
    )
    assert (oracle.accepted, oracle.eligible) == (True, None)
    assert (calibration.accepted, calibration.eligible) == (True, None)
    assert (parity.accepted, parity.eligible) == (True, None)
    assert (candidate.accepted, candidate.eligible) == (True, True)


@pytest.mark.parametrize(
    "trials,cleanup,parity",
    [
        ([_trial("known-good", 1.0) for _ in range(3)], _cleanup(), None),
        (
            [_trial("known-good", 1.0) for _ in range(3)]
            + [
                _trial(
                    "known-good",
                    1.0,
                    retained=False,
                    reward=None,
                    completed=False,
                    provenance_matches=False,
                    infrastructure_failure="lost result",
                )
            ],
            _cleanup(),
            None,
        ),
        ([_trial("known-good", 1.0, reward=0.0) for _ in range(4)], _cleanup(), None),
        (
            [_trial("known-good", 1.0) for _ in range(3)]
            + [
                _trial(
                    "known-good",
                    1.0,
                    provenance_matches=False,
                    package_sha256="e" * 64,
                )
            ],
            _cleanup(),
            None,
        ),
        ([_trial("known-good", 1.0) for _ in range(4)], _cleanup(passed=False), None),
    ],
)
def test_candidate_gate_rejects_incomplete_wrong_or_heterogeneous_evidence(
    trials: list[Ec2Trial], cleanup: CleanupSummary, parity: bool | None
) -> None:
    gate = candidate_gate(
        run_id="run-1",
        purpose="candidate_gate",
        task_source="static",
        execution=_execution(attempts=4, concurrency=4),
        benchmark=_benchmark(),
        execution_package=_package(),
        candidate=_candidate(),
        trials=trials,
        cleanup=cleanup,
        local_ec2_parity=parity,
    )
    assert (gate.accepted, gate.eligible) == (False, False)
    assert gate.failure_reasons


def test_parity_mismatch_is_reported_not_eligible() -> None:
    gate = candidate_gate(
        run_id="run-1",
        purpose="parity",
        task_source="taskify",
        execution=_execution(source="taskify"),
        benchmark=_benchmark("taskify"),
        execution_package=_package(),
        candidate=None,
        trials=[_trial("oracle", 1.0), _trial("known-good", 1.0), _trial("scripted-bad", 0.0)],
        cleanup=_cleanup(),
        local_ec2_parity=False,
    )
    assert (gate.accepted, gate.eligible) == (False, None)
    assert "local EC2 parity did not match" in gate.failure_reasons


def test_candidate_provenance_cannot_claim_a_mismatch() -> None:
    data = _trial("known-good", 1.0).model_dump()
    data["package_sha256"] = "e" * 64
    with pytest.raises(ValidationError, match="provenance match claim"):
        candidate_gate(
            run_id="run-1",
            purpose="candidate_gate",
            task_source="static",
            execution=_execution(),
            benchmark=_benchmark(),
            execution_package=_package(),
            candidate=_candidate(),
            trials=[Ec2Trial.model_validate(data)],
            cleanup=_cleanup(),
            local_ec2_parity=None,
        )


def _taskify_artifact(tmp_path: Path) -> Path:
    evidence = FailureEvidence.model_validate_json((FIXTURES / "failure.json").read_text())
    trace = json.loads((FIXTURES / "failing_trace.json").read_text())
    scenario = build_scenario(evidence, trace)
    artifact = tmp_path / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "scenario.yaml").write_text(dump_scenario_yaml(scenario))
    task = artifact / "harbor"
    task.mkdir()
    (task / "task.toml").write_text("name = 'synthetic'\n")
    task_digest = harbor_task_sha256(task)
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "scenario_id": scenario.id,
                "scenario": {"path": "scenario.yaml", "sha256": scenario_sha256(scenario)},
            }
        )
    )
    report = ReproductionReport(
        status="VALIDATED",
        replay_mode="calibration",
        source_kind="trace",
        source=SourceProvenance(
            source_revision="a" * 40,
            trace_id="trace",
            session_id="session",
            agent_config_fingerprint="b" * 64,
        ),
        calibration=CalibrationIdentity(
            kind="source_replay",
            model="scripted/bad",
            agent_config_fingerprint="b" * 64,
            expected_tools=["get_invoice", "refund_invoice"],
            derived_from_observed_failure=False,
        ),
        execution_package=_package(),
        harbor_task=HarborTaskIdentity(sha256=task_digest),
        attempts=1,
        failure_count=1,
        trials=[
            TrialResult(name="oracle", attempted=True, reward=1.0),
            TrialResult(name="known-good", attempted=True, reward=1.0),
            TrialResult(name="originating", attempted=True, reward=0.0),
        ],
    )
    (artifact / "reproduction.json").write_text(report.model_dump_json())
    return artifact


def test_taskify_integrity_binds_manifest_scenario_and_rendered_task(tmp_path: Path) -> None:
    artifact = _taskify_artifact(tmp_path)
    task, report, identity = run_harbor_ec2.resolve_task("taskify", artifact)
    assert report is not None and identity.scenario_id
    assert task == (artifact / "harbor").resolve()

    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest["scenario"]["sha256"] = "0" * 64
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(run_harbor_ec2.Ec2RunError, match="Scenario SHA-256"):
        run_harbor_ec2.resolve_task("taskify", artifact)

    artifact = _taskify_artifact(tmp_path / "fresh")
    (artifact / "harbor" / "task.toml").write_text("name = 'tampered'\n")
    with pytest.raises(run_harbor_ec2.Ec2RunError, match="SHA-256"):
        run_harbor_ec2.resolve_task("taskify", artifact)


def test_task_tree_hash_is_deterministic_and_rejects_symlinks(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "b").write_text("two")
    (task / "a").write_text("one")
    first = harbor_task_sha256(task)
    assert first == harbor_task_sha256(task)
    (task / "a").write_text("changed")
    assert first != harbor_task_sha256(task)
    (task / "link").symlink_to(task / "a")
    with pytest.raises(ValueError, match="symlinks"):
        harbor_task_sha256(task)


def test_cli_returns_nonzero_after_a_written_rejected_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "candidate-gate.json"
    gate = candidate_gate(
        run_id="run-1",
        purpose="oracle_smoke",
        task_source="static",
        execution=_execution(),
        benchmark=_benchmark(),
        execution_package=_package(),
        candidate=None,
        trials=[_trial("oracle", 1.0, reward=0.0)],
        cleanup=_cleanup(),
        local_ec2_parity=None,
    )
    report_path.write_text(gate.model_dump_json())
    monkeypatch.setattr(
        run_harbor_ec2, "run", lambda **_: run_harbor_ec2.RunOutcome(report_path, gate)
    )
    assert run_harbor_ec2.main(["--kind", "oracle"]) == 1
    assert report_path.is_file()


def test_phase6_demo_assertion_requires_accepted_parity_and_scale(tmp_path: Path) -> None:
    parity = candidate_gate(
        run_id="parity",
        purpose="parity",
        task_source="taskify",
        execution=_execution(source="taskify"),
        benchmark=_benchmark("taskify"),
        execution_package=_package(),
        candidate=None,
        trials=[_trial("oracle", 1.0), _trial("known-good", 1.0), _trial("scripted-bad", 0.0)],
        cleanup=_cleanup(),
        local_ec2_parity=True,
    )
    scale = candidate_gate(
        run_id="scale",
        purpose="candidate_gate",
        task_source="static",
        execution=_execution(attempts=4, concurrency=4),
        benchmark=_benchmark(),
        execution_package=_package(),
        candidate=_candidate(),
        trials=[_trial("known-good", 1.0) for _ in range(4)],
        cleanup=_cleanup(),
        local_ec2_parity=None,
    )
    parity_path, scale_path = tmp_path / "parity.json", tmp_path / "scale.json"
    parity_path.write_text(parity.model_dump_json())
    scale_path.write_text(scale.model_dump_json())
    assert_phase6_acceptance.assert_acceptance(parity_path, scale_path)
