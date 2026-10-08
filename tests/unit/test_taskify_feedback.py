"""Focused Phase 5 feedback regressions that require no cloud or DSQL state."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import agentops_demo.taskify.harbor_runner as harbor_runner
from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.contracts.scenario import Scenario
from agentops_demo.harbor.provenance import PackageProvenance
from agentops_demo.taskify.contracts import FailureEvidence
from agentops_demo.taskify.harbor_renderer import render_harbor_task
from agentops_demo.taskify.reproduction import (
    CalibrationIdentity,
    ReproductionReport,
    SourceProvenance,
    TrialResult,
)
from agentops_demo.taskify.scenario_builder import (
    ScenarioBuildError,
    UnsupportedEvaluatorError,
    UnsupportedFailureSubtypeError,
    build_scenario,
)
from agentops_demo.taskify.trace import (
    TraceCorrelationError,
    WorldSnapshotError,
    correlate_trace,
    ordered_successful_tool_spans,
    snapshot_from_root,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/taskify"


def _fixture() -> tuple[FailureEvidence, dict[str, object]]:
    evidence = FailureEvidence.model_validate_json((FIXTURES / "failure.json").read_text())
    trace = json.loads((FIXTURES / "failing_trace.json").read_text())
    return evidence, trace


def test_tool_order_is_numeric_then_span_id() -> None:
    _, trace = _fixture()
    spans = trace["evaluationInput"]["sessionSpans"]
    tools = [span for span in spans if span["name"] in {"get_invoice", "refund_invoice"}]
    tools[0]["startTimeUnixNano"] = "20"
    tools[0]["spanId"] = "z"
    tools[1]["startTimeUnixNano"] = "3"
    tools[1]["spanId"] = "a"
    assert [name for name, _ in ordered_successful_tool_spans(tools)] == [
        "refund_invoice",
        "get_invoice",
    ]


@pytest.mark.parametrize("timestamp", [None, "bad", "-1", True])
def test_tool_order_rejects_invalid_timestamps(timestamp: object) -> None:
    _, trace = _fixture()
    tool = next(
        span
        for span in trace["evaluationInput"]["sessionSpans"]
        if span["name"] in {"get_invoice", "refund_invoice"}
    )
    tool["startTimeUnixNano"] = timestamp
    with pytest.raises(TraceCorrelationError):
        ordered_successful_tool_spans([tool])


def test_non_refund_failure_subtype_is_rejected() -> None:
    evidence, trace = _fixture()
    trace = copy.deepcopy(trace)
    trace["evaluationInput"]["sessionSpans"] = [
        span
        for span in trace["evaluationInput"]["sessionSpans"]
        if span["name"] != "refund_invoice"
    ]
    with pytest.raises(UnsupportedFailureSubtypeError):
        build_scenario(evidence, trace)


def test_renderer_oracle_uses_scenario_invoice_not_static_value(tmp_path: Path) -> None:
    evidence, trace = _fixture()
    raw = build_scenario(evidence, trace).model_dump(mode="json")
    serialized = json.dumps(raw).replace("inv-123", "inv-456")
    scenario = Scenario.model_validate_json(serialized)
    output = asyncio.run(render_harbor_task(scenario, tmp_path / "task"))
    oracle = (output / "solution/solve.sh").read_text()
    hidden = (output / "tests/scenario.json").read_text()
    seed = (output / "environment/seed.sql").read_text()
    assert "inv-456" in oracle and "inv-123" not in oracle
    assert "inv-456" in hidden and "inv-123" not in hidden
    assert "inv-456" in seed and "inv-123" not in seed


def _source() -> SourceProvenance:
    return SourceProvenance(
        source_revision="b" * 40,
        trace_id="trace",
        session_id="session",
        agent_config_fingerprint="a" * 64,
    )


def _calibration() -> CalibrationIdentity:
    return CalibrationIdentity(
        kind="source_replay",
        model="scripted/bad",
        agent_config_fingerprint="a" * 64,
        expected_tools=["get_invoice", "refund_invoice"],
        derived_from_observed_failure=False,
    )


def _package(revision: str = "c" * 40) -> PackageProvenance:
    return PackageProvenance(
        source_revision=revision,
        filename="candidate.whl",
        sha256="d" * 64,
        worktree_clean=True,
    )


def test_bedrock_disputed_refund_derives_distinct_scripted_calibration() -> None:
    evidence, trace = _fixture()
    scenario = build_scenario(evidence, trace)
    source = scenario.provenance.originating_agent_config
    assert source is not None
    raw = source.model_dump(mode="json")
    raw["model"] = {"provider": "bedrock", "model_id": "unit-test"}
    source = AgentConfig.model_validate(raw)
    selected = harbor_runner._calibration(source, ["get_invoice", "refund_invoice"])
    assert not isinstance(selected, str)
    calibration, identity = selected
    assert source.model.provider == "bedrock"
    assert calibration.model.provider == "scripted"
    assert calibration.model.model_id == "bad"
    assert identity.kind == "trajectory_calibration"
    assert identity.derived_from_observed_failure is True
    assert identity.agent_config_fingerprint != source.fingerprint()


def test_unsupported_bedrock_trajectory_never_derives_calibration() -> None:
    evidence, trace = _fixture()
    scenario = build_scenario(evidence, trace)
    source = scenario.provenance.originating_agent_config
    assert source is not None
    raw = source.model_dump(mode="json")
    raw["model"] = {"provider": "bedrock", "model_id": "unit-test"}
    source = AgentConfig.model_validate(raw)
    assert isinstance(harbor_runner._calibration(source, ["get_invoice"]), str)


@pytest.mark.parametrize(
    "trials,failures",
    [
        ([], 0),
        ([TrialResult(name="oracle", attempted=True, reward=0.0)], 0),
        (
            [
                TrialResult(name="oracle", attempted=True, reward=1.0),
                TrialResult(name="known-good", attempted=True, reward=0.0),
                TrialResult(name="originating", attempted=True, reward=0.0),
            ],
            1,
        ),
        (
            [
                TrialResult(name="oracle", attempted=True, reward=1.0),
                TrialResult(name="known-good", attempted=True, reward=1.0),
                TrialResult(name="originating", attempted=True, reward=1.0),
            ],
            0,
        ),
    ],
)
def test_reproduction_contract_rejects_invalid_validated(
    trials: list[TrialResult], failures: int
) -> None:
    with pytest.raises(ValidationError):
        ReproductionReport(
            status="VALIDATED",
            replay_mode="calibration",
            source_kind="trace",
            source=_source(),
            execution_package=_package(),
            attempts=int(any(trial.name == "originating" for trial in trials)),
            failure_count=failures,
            trials=trials,
        )


def test_failure_evidence_accepts_producer_trace_span_count() -> None:
    evidence, _ = _fixture()
    raw = evidence.model_dump(mode="json")
    raw["trace_span_count"] = 47
    assert FailureEvidence.model_validate(raw).trace_span_count == 47


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("traceId", "f" * 32),
        ("session.id", "wrong-session"),
        ("agentops.source_revision", "c" * 40),
        ("agentops.model_provider", "bedrock"),
        ("agentops.model_id", "wrong"),
        ("agentops.prompt_version", "wrong"),
        ("agentops.tools_version", "wrong"),
        ("agentops.harness_framework", "wrong"),
        ("agentops.harness_version", "wrong"),
        ("agentops.agent_config_fingerprint", "0" * 64),
        ("agentops.invocation.instruction", "wrong instruction"),
    ],
)
def test_trace_correlation_rejects_each_mismatch(field: str, value: str) -> None:
    evidence, trace = _fixture()
    trace = copy.deepcopy(trace)
    root = trace["evaluationInput"]["sessionSpans"][0]
    if field == "traceId":
        root[field] = value
    elif field == "agentops.agent_config_fingerprint":
        raw = evidence.model_dump(mode="json")
        raw["candidate"]["agent_config_fingerprint"] = value
        evidence = FailureEvidence.model_validate(raw)
    else:
        root["attributes"][field] = value
    with pytest.raises(TraceCorrelationError):
        correlate_trace(evidence, trace)


def test_trace_correlation_rejects_missing_scope_and_provenance() -> None:
    evidence, trace = _fixture()
    no_scope = copy.deepcopy(trace)
    for span in no_scope["evaluationInput"]["sessionSpans"]:
        span.pop("scope", None)
    with pytest.raises(TraceCorrelationError, match="supported OpenInference scope"):
        correlate_trace(evidence, no_scope)
    missing = copy.deepcopy(trace)
    missing["evaluationInput"]["sessionSpans"][0]["attributes"].pop("agentops.model_id")
    with pytest.raises(TraceCorrelationError, match="incomplete candidate provenance"):
        correlate_trace(evidence, missing)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda root: root.pop("agentops.world_state.before"),
        lambda root: root.__setitem__("agentops.world_state.before", "not-json"),
        lambda root: root.__setitem__("agentops.world_state.before", '{"schema_version":"2"}'),
        lambda root: root.__setitem__("agentops.world_snapshot.schema_version", "2"),
        lambda root: root.__setitem__("agentops.world_snapshot.kind", "wrong"),
        lambda root: root.__setitem__("agentops.world_snapshot.sha256", "0" * 64),
    ],
)
def test_snapshot_rejects_incomplete_or_mismatched_evidence(mutation: object) -> None:
    _, trace = _fixture()
    root = copy.deepcopy(trace)["evaluationInput"]["sessionSpans"][0]["attributes"]
    mutation(root)  # type: ignore[operator]
    with pytest.raises(WorldSnapshotError):
        snapshot_from_root(root)


def test_scenario_builder_rejects_unknown_tools_and_malformed_refunds() -> None:
    evidence, trace = _fixture()
    unknown = copy.deepcopy(trace)
    unknown["evaluationInput"]["sessionSpans"][2]["name"] = "delete_invoice"
    unknown["evaluationInput"]["sessionSpans"][2]["attributes"]["tool.name"] = "delete_invoice"
    with pytest.raises(ScenarioBuildError, match="unsupported tool"):
        build_scenario(evidence, unknown)
    malformed = copy.deepcopy(trace)
    malformed["evaluationInput"]["sessionSpans"][2]["attributes"]["input.value"] = "[]"
    with pytest.raises(ScenarioBuildError, match="not an object"):
        build_scenario(evidence, malformed)
    missing_id = copy.deepcopy(trace)
    missing_id["evaluationInput"]["sessionSpans"][2]["attributes"]["input.value"] = '{"reason":"x"}'
    with pytest.raises(ScenarioBuildError, match="missing invoice_id"):
        build_scenario(evidence, missing_id)
    missing_reason = copy.deepcopy(trace)
    missing_reason["evaluationInput"]["sessionSpans"][2]["attributes"]["input.value"] = (
        '{"invoice_id":"inv-123"}'
    )
    with pytest.raises(ScenarioBuildError, match="missing reason"):
        build_scenario(evidence, missing_reason)
    multiple = copy.deepcopy(trace)
    refund = copy.deepcopy(multiple["evaluationInput"]["sessionSpans"][2])
    refund["spanId"] = "0000000000000004"
    refund["startTimeUnixNano"] = "3"
    refund["attributes"]["input.value"] = '{"invoice_id":"inv-456","reason":"x"}'
    multiple["evaluationInput"]["sessionSpans"].append(refund)
    with pytest.raises(ScenarioBuildError, match="multiple refunded"):
        build_scenario(evidence, multiple)
    evaluator_raw = evidence.model_dump(mode="json")
    evaluator_raw["evaluator"] = {"id": "id", "name": "unknown", "label": "FAIL", "value": 0.0}
    unknown_evaluator = FailureEvidence.model_validate(evaluator_raw)
    with pytest.raises(UnsupportedEvaluatorError):
        build_scenario(unknown_evaluator, trace)


@pytest.mark.parametrize(
    ("status", "trials", "attempts", "failures", "reason"),
    [
        ("NOT_REPRODUCED", [], 0, 0, None),
        ("UNSUPPORTED", [], 0, 0, ""),
        ("UNSUPPORTED", [TrialResult(name="originating", attempted=True)], 1, 0, "unsupported"),
    ],
)
def test_reproduction_contract_rejects_invalid_status_semantics(
    status: str,
    trials: list[TrialResult],
    attempts: int,
    failures: int,
    reason: str | None,
) -> None:
    with pytest.raises(ValidationError):
        ReproductionReport(
            status=status,
            replay_mode="calibration",
            source_kind="trace",
            source=_source(),
            attempts=attempts,
            failure_count=failures,
            trials=trials,
            reason=reason,
        )


def test_reproduction_contract_distinguishes_exact_and_calibration_revisions() -> None:
    source = _source()
    with pytest.raises(ValidationError, match="package revision"):
        ReproductionReport(
            status="CANDIDATE",
            replay_mode="exact_source",
            source_kind="trace",
            source=source,
            calibration=_calibration(),
            execution_package=_package(),
        )
    report = ReproductionReport(
        status="CANDIDATE",
        replay_mode="calibration",
        source_kind="trace",
        source=source,
        calibration=_calibration(),
        execution_package=_package(),
    )
    assert report.execution_package is not None


def _mock_reproduction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rewards: dict[str, float | None]
) -> Path:
    evidence, trace = _fixture()
    scenario = build_scenario(evidence, trace)
    scenario_path = tmp_path / "scenario.yaml"
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"wheel")
    package = PackageProvenance(
        source_revision="c" * 40,
        filename=wheel.name,
        sha256="d" * 64,
        worktree_clean=True,
    )

    monkeypatch.setattr(harbor_runner, "load_scenario", lambda _: scenario)
    monkeypatch.setattr(harbor_runner, "build_clean_wheel", lambda _: (wheel, package))
    monkeypatch.setattr(
        harbor_runner, "_run_harbor", lambda **kwargs: kwargs["jobs_root"] / kwargs["name"]
    )

    async def render(*_: object) -> Path:
        task = tmp_path / "harbor"
        task.mkdir(exist_ok=True)
        (task / "task.toml").write_text("name = 'synthetic'\n")
        return task

    monkeypatch.setattr(harbor_runner, "render_harbor_task", render)
    monkeypatch.setattr(
        harbor_runner, "validate_runtime_provenance", lambda *_args, **_kwargs: None
    )

    def read(name: str, job_dir: Path) -> TrialResult:
        if rewards[name] is None:
            raise ValueError("synthetic infrastructure failure")
        tools = {
            "oracle": [],
            "known-good": ["get_invoice", "escalate_dispute"],
            "originating": ["get_invoice", "refund_invoice"],
        }[name]
        return TrialResult(
            name=name, attempted=True, reward=rewards[name], tool_calls=tools, job_dir=str(job_dir)
        )

    monkeypatch.setattr(harbor_runner, "read_trial_result", read)
    return scenario_path


def test_runner_retains_known_good_failure_before_expectation_assertion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = harbor_runner.reproduce(
        _mock_reproduction(
            monkeypatch, tmp_path, {"oracle": 1.0, "known-good": 0.0, "originating": 0.0}
        )
    )
    assert report.status == "NEEDS_REVIEW"
    assert report.attempts == report.failure_count == 0
    assert report.trials[-1].name == "known-good"
    assert report.trials[-1].reward == 0.0


def test_runner_retains_source_non_reproduction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = harbor_runner.reproduce(
        _mock_reproduction(
            monkeypatch, tmp_path, {"oracle": 1.0, "known-good": 1.0, "originating": 1.0}
        )
    )
    assert report.status == "NOT_REPRODUCED"
    assert (report.attempts, report.failure_count) == (1, 0)
    assert report.trials[-1].reward == 1.0


def test_runner_retains_source_infrastructure_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = harbor_runner.reproduce(
        _mock_reproduction(
            monkeypatch, tmp_path, {"oracle": 1.0, "known-good": 1.0, "originating": None}
        )
    )
    assert report.status == "NEEDS_REVIEW"
    assert (report.attempts, report.failure_count) == (1, 0)
    assert report.trials[-1].reward is None


def test_runner_rejects_unsupported_source_before_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence, trace = _fixture()
    raw = build_scenario(evidence, trace).model_dump(mode="json")
    raw["provenance"]["originating_agent_config"]["model"] = {
        "provider": "bedrock",
        "model_id": "model",
    }
    raw["observed_failure"]["tool_calls"] = raw["observed_failure"]["tool_calls"][:1]
    scenario = Scenario.model_validate(raw)
    scenario_path = tmp_path / "scenario.yaml"
    monkeypatch.setattr(harbor_runner, "load_scenario", lambda _: scenario)
    monkeypatch.setattr(
        harbor_runner,
        "_run_harbor",
        lambda **_: pytest.fail("unsupported source must not run Harbor controls"),
    )
    monkeypatch.setattr(
        harbor_runner,
        "build_clean_wheel",
        lambda _: pytest.fail("unsupported source must not build an execution package"),
    )
    report = harbor_runner.reproduce(scenario_path)
    assert report.status == "UNSUPPORTED"
    assert report.execution_package is None
    assert report.calibration is None
    assert report.trials == []
    assert report.attempts == report.failure_count == 0
    assert report.reason
