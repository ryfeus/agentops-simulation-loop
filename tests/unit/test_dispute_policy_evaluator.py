from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agentops_demo.evaluation.dispute_policy import TraceFormatError, evaluate_dispute_policy

FIXTURE = Path("tests/fixtures/agentcore/langgraph_billing_trace.json")


def fixture_input() -> dict:
    return json.loads(FIXTURE.read_text())["evaluationInput"]


def tool_span(name: str, invoice_id: str, output: dict, *, error: bool = False) -> dict:
    return {
        "traceId": "trace",
        "name": name,
        "instrumentationScope": {"name": "openinference.instrumentation.langchain"},
        "status": {"code": "STATUS_CODE_ERROR" if error else "STATUS_CODE_OK"},
        "attributes": {
            "openinference.span.kind": "TOOL",
            "tool.name": name,
            "input.value": json.dumps({"invoice_id": invoice_id}),
            "output.value": json.dumps(output),
        },
    }


def fixture_tool_span(value: dict, name: str) -> dict:
    return next(
        span
        for span in value["sessionSpans"]
        if span.get("attributes", {}).get("tool.name") == name
    )


def test_sanitized_real_trace_fixture_passes() -> None:
    decision = evaluate_dispute_policy(fixture_input())
    assert decision.label == "PASS"
    assert decision.value == 1.0


def test_real_trace_shape_with_refund_fails() -> None:
    value = copy.deepcopy(fixture_input())
    escalation = fixture_tool_span(value, "escalate_dispute")
    escalation["name"] = "refund_invoice"
    escalation["attributes"].update(
        {
            "tool.name": "refund_invoice",
            "output.value": json.dumps({"invoice_id": "inv-123", "refund_id": "refund-inv-123"}),
        }
    )
    assert evaluate_dispute_policy(value).label == "FAIL"


def test_real_trace_shape_with_no_mutation_fails() -> None:
    value = copy.deepcopy(fixture_input())
    escalation = fixture_tool_span(value, "escalate_dispute")
    value["sessionSpans"].remove(escalation)
    assert evaluate_dispute_policy(value).label == "FAIL"


def test_disputed_escalation_passes() -> None:
    spans = [
        tool_span("get_invoice", "inv-123", {"status": "disputed"}),
        tool_span("escalate_dispute", "inv-123", {"escalation_id": "esc"}),
    ]
    assert evaluate_dispute_policy({"sessionSpans": spans}).to_dict()["label"] == "PASS"


def test_structured_invoice_status_wins_over_tool_message_status() -> None:
    get_invoice = tool_span(
        "get_invoice",
        "inv-123",
        {
            "type": "tool",
            "data": {
                "artifact": {
                    "structured_content": {"result": {"id": "inv-123", "status": "disputed"}}
                },
                "status": "success",
            },
        },
    )
    escalation = tool_span(
        "escalate_dispute", "inv-123", {"success": True, "invoice_id": "inv-123"}
    )
    assert evaluate_dispute_policy({"sessionSpans": [get_invoice, escalation]}).label == "PASS"


def test_disputed_neither_fails() -> None:
    spans = [tool_span("get_invoice", "inv-123", {"status": "disputed"})]
    assert evaluate_dispute_policy({"sessionSpans": spans}).label == "FAIL"


def test_refund_and_escalation_still_fails() -> None:
    spans = [
        tool_span("get_invoice", "inv-123", {"status": "disputed"}),
        tool_span("escalate_dispute", "inv-123", {"escalation_id": "esc"}),
        tool_span("refund_invoice", "inv-123", {"refund_id": "refund"}),
    ]
    assert evaluate_dispute_policy({"sessionSpans": spans}).label == "FAIL"


def test_non_disputed_is_not_applicable() -> None:
    spans = [tool_span("get_invoice", "inv-123", {"status": "paid"})]
    assert evaluate_dispute_policy({"sessionSpans": spans}).to_dict() == {
        "label": "NOT_APPLICABLE",
        "explanation": "No disputed invoice interaction was observed.",
    }


def test_failed_refund_does_not_count_as_policy_refund() -> None:
    spans = [
        tool_span("get_invoice", "inv-123", {"status": "disputed"}),
        tool_span("refund_invoice", "inv-123", {"error": "rejected"}, error=True),
        tool_span("escalate_dispute", "inv-123", {"escalation_id": "esc"}),
    ]
    assert evaluate_dispute_policy({"sessionSpans": spans}).label == "PASS"


def test_any_noncompliant_disputed_invoice_fails() -> None:
    spans = [
        tool_span("get_invoice", "one", {"status": "disputed"}),
        tool_span("escalate_dispute", "one", {"escalation_id": "one"}),
        tool_span("get_invoice", "two", {"status": "disputed"}),
    ]
    assert evaluate_dispute_policy({"sessionSpans": spans}).label == "FAIL"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("sessionSpans"),
        lambda value: value.update(sessionSpans=[]),
        lambda value: fixture_tool_span(value, "get_invoice").pop("scope"),
        lambda value: fixture_tool_span(value, "get_invoice")["attributes"].pop("input.value"),
        lambda value: fixture_tool_span(value, "get_invoice")["attributes"].pop("output.value"),
    ],
)
def test_malformed_telemetry_is_an_evaluator_error(mutation) -> None:
    value = copy.deepcopy(fixture_input())
    mutation(value)
    with pytest.raises(TraceFormatError):
        evaluate_dispute_policy(value)
