from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

FIXTURE = Path("tests/fixtures/agentcore/langgraph_billing_trace.json")


def load_handler():
    spec = importlib.util.spec_from_file_location(
        "evaluator_handler", "deploy/evaluator/handler.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.handler


def event() -> dict:
    return json.loads(FIXTURE.read_text())


def test_lambda_returns_agentcore_success_envelope() -> None:
    assert load_handler()(event(), None)["label"] == "PASS"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schemaVersion", "2.0"),
        ("evaluatorName", "Other"),
        ("evaluatorId", "Other-ABCDEFGHIJ"),
        ("evaluationLevel", "SESSION"),
        ("evaluationInput", None),
        ("evaluationTarget", None),
    ],
)
def test_lambda_rejects_invalid_envelope(field: str, value: object) -> None:
    payload = event()
    payload[field] = value
    result = load_handler()(payload, None)
    assert result["errorCode"] == "INVALID_EVALUATION_REQUEST"


@pytest.mark.parametrize("missing", ["evaluatorId", "evaluatorName"])
def test_lambda_rejects_partial_evaluator_identity(missing: str) -> None:
    payload = event()
    payload.pop(missing)
    assert load_handler()(payload, None)["errorCode"] == "INVALID_EVALUATION_REQUEST"


def test_lambda_accepts_observed_identity_less_aws_envelope() -> None:
    payload = event()
    payload.pop("evaluatorId")
    payload.pop("evaluatorName")
    assert load_handler()(payload, None)["label"] == "PASS"


def test_lambda_distinguishes_trace_format_error() -> None:
    payload = copy.deepcopy(event())
    tool_span = next(
        span
        for span in payload["evaluationInput"]["sessionSpans"]
        if span["attributes"].get("tool.name") == "get_invoice"
    )
    tool_span["attributes"].pop("output.value")
    assert load_handler()(payload, None)["errorCode"] == "TRACE_FORMAT_UNSUPPORTED"
