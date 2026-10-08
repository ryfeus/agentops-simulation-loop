from __future__ import annotations

import json
from pathlib import Path

from agentops_demo.evaluation.suite import (
    evaluate_mutation_cardinality,
    evaluate_tool_workflow,
)

FIXTURE = Path("tests/fixtures/agentcore/langgraph_billing_trace.json")


def _payload() -> dict:
    return json.loads(FIXTURE.read_text())["evaluationInput"]


def test_online_safe_evaluators_accept_valid_inspected_trajectory() -> None:
    assert evaluate_mutation_cardinality(_payload()).label == "PASS"
    assert evaluate_tool_workflow(_payload()).label == "PASS"


def test_cardinality_detects_repeated_write_attempts() -> None:
    payload = _payload()
    original = next(
        span
        for span in payload["sessionSpans"]
        if span.get("attributes", {}).get("tool.name") == "escalate_dispute"
    )
    duplicate = dict(original)
    duplicate["spanId"] = "duplicate"
    payload["sessionSpans"].append(duplicate)

    assert evaluate_mutation_cardinality(payload).label == "FAIL"


def test_workflow_detects_write_before_matching_inspection() -> None:
    payload = _payload()
    payload["sessionSpans"] = list(reversed(payload["sessionSpans"]))

    assert evaluate_tool_workflow(payload).label == "FAIL"
