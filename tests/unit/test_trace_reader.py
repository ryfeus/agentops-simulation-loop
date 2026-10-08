from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentops_demo.evaluation.trace_reader import (
    parse_json_record,
    spans_for_session,
    validate_observable_trace,
)

FIXTURE = Path("tests/fixtures/agentcore/langgraph_billing_trace.json")


def spans() -> list[dict]:
    return json.loads(FIXTURE.read_text())["evaluationInput"]["sessionSpans"]


def test_parse_nested_cloudwatch_record() -> None:
    assert parse_json_record(json.dumps({"@message": json.dumps({"traceId": "trace"})})) == {
        "traceId": "trace"
    }


def test_reader_accepts_cloudwatch_scope_field() -> None:
    live_shape = [
        {
            "traceId": "trace-live",
            "name": "get_invoice",
            "scope": {"name": "openinference.instrumentation.langchain"},
            "attributes": {
                "session.id": "session-live",
                "output.value": '{"status":"disputed"}',
                "agentops.source_revision": "revision-live",
            },
        }
    ]
    trace_id, selected = validate_observable_trace(
        live_shape,
        session_id="session-live",
        expected={"agentops.source_revision": "revision-live"},
    )
    assert trace_id == "trace-live"
    assert selected == live_shape


def test_exact_session_selects_trace() -> None:
    trace_id, selected = spans_for_session(spans(), "session-sanitized")
    assert trace_id == "00000000000000000000000000000001"
    assert len(selected) == 12


def test_observable_trace_requires_scope_tool_and_provenance() -> None:
    expected = {
        "agentops.source_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "agentops.agent_config_fingerprint": "a" * 64,
    }
    assert validate_observable_trace(spans(), session_id="session-sanitized", expected=expected)[
        0
    ].endswith("1")
    with pytest.raises(ValueError, match="provenance"):
        validate_observable_trace(
            spans(), session_id="session-sanitized", expected={"agentops.model_id": "other"}
        )
