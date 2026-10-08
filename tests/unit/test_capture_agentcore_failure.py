from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from agentops_demo.agentcore.candidates import resolve_candidate
from agentops_demo.contracts.agent_config import AgentConfig
from scripts.capture_agentcore_failure import build_failure_evidence

FIXTURES = Path("tests/fixtures/taskify")


def _trace() -> dict[str, object]:
    return json.loads((FIXTURES / "failing_trace.json").read_text())


def _evaluation() -> dict[str, object]:
    return {
        "evaluator": "DisputePolicyCompliance",
        "evaluator_id": "DisputePolicyCompliance-TEST",
        "label": "FAIL",
        "value": 0.0,
        "trace_id": "00000000000000000000000000000005",
        "session_id": "session-taskify-sanitized",
        "timestamp": "0",
        "explanation": "disputed invoice was refunded",
    }


def _build(*, evaluation: dict[str, object] | None = None, trace: dict[str, object] | None = None):
    return build_failure_evidence(
        evaluation=evaluation or _evaluation(),
        trace=trace or _trace(),
        session_id="session-taskify-sanitized",
        trace_id="00000000000000000000000000000005",
        service_name="agentops_demo_dev.DEFAULT",
        trace_log_group="/aws/bedrock-agentcore/runtimes/sanitized",
        result_log_group="/aws/bedrock-agentcore/evaluations/sanitized",
    )


def test_capture_builds_existing_failure_contract_from_correlated_trace() -> None:
    evidence = _build()
    assert evidence["source"]["instruction"] == (
        "I was charged twice for April. Please refund invoice inv-123."
    )
    assert evidence["trajectory"] == ["get_invoice", "refund_invoice"]
    assert evidence["trace_span_count"] == 3


def test_capture_rejects_wrong_evaluation_identity() -> None:
    evaluation = _evaluation()
    evaluation["session_id"] = "other-session"
    with pytest.raises(ValueError, match="does not match"):
        _build(evaluation=evaluation)


def test_capture_rejects_non_failure_evaluation() -> None:
    evaluation = _evaluation()
    evaluation["label"] = "PASS"
    evaluation["value"] = 1.0
    with pytest.raises(ValueError, match="not a FAIL"):
        _build(evaluation=evaluation)


def test_capture_requires_trace_instruction() -> None:
    trace = deepcopy(_trace())
    root = trace["evaluationInput"]["sessionSpans"][0]
    del root["attributes"]["agentops.invocation.instruction"]
    with pytest.raises(ValueError, match="missing invocation instruction"):
        _build(trace=trace)


def test_capture_uses_candidate_identity_from_the_trace() -> None:
    trace = deepcopy(_trace())
    root = trace["evaluationInput"]["sessionSpans"][0]
    deployed = root["attributes"]
    config = resolve_candidate(
        AgentConfig.model_validate(
            {
                "agent": {"source_revision": deployed["agentops.source_revision"]},
                "model": {
                    "provider": deployed["agentops.model_provider"],
                    "model_id": deployed["agentops.model_id"],
                },
                "prompt": {"version": deployed["agentops.prompt_version"]},
                "tools": {"version": deployed["agentops.tools_version"]},
                "harness": {
                    "framework": deployed["agentops.harness_framework"],
                    "version": deployed["agentops.harness_version"],
                },
            }
        ),
        "scripted-bad",
    )
    deployed.update(
        {
            "agentops.model_provider": config.model.provider,
            "agentops.model_id": config.model.model_id,
            "agentops.agent_config_fingerprint": config.fingerprint(),
        }
    )

    evidence = _build(trace=trace)
    assert evidence["candidate"]["model_provider"] == "scripted"
    assert evidence["candidate"]["model_id"] == "bad"
    assert evidence["candidate"]["agent_config_fingerprint"] == config.fingerprint()
