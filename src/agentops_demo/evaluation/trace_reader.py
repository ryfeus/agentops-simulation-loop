"""Strict helpers for AgentCore OpenTelemetry span and result records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from agentops_demo.evaluation.dispute_policy import SUPPORTED_SCOPE


def parse_json_record(value: str | Mapping[str, Any]) -> dict[str, Any]:
    """Decode one CloudWatch message, including common nested message envelopes."""

    current: Any = value
    for _ in range(4):
        if isinstance(current, str):
            try:
                current = json.loads(current)
            except json.JSONDecodeError as exc:
                raise ValueError("CloudWatch record is not valid JSON") from exc
        elif isinstance(current, Mapping) and isinstance(current.get("@message"), str):
            current = current["@message"]
        elif isinstance(current, Mapping) and isinstance(current.get("message"), str):
            nested = current["message"]
            try:
                current = json.loads(nested)
            except json.JSONDecodeError:
                break
        else:
            break
    if not isinstance(current, Mapping):
        raise ValueError("CloudWatch record must decode to an object")
    return dict(current)


def span_trace_id(span: Mapping[str, Any]) -> str | None:
    value = span.get("traceId", span.get("trace_id"))
    return value if isinstance(value, str) and value else None


def span_attributes(span: Mapping[str, Any]) -> Mapping[str, Any]:
    value = span.get("attributes", {})
    return value if isinstance(value, Mapping) else {}


def span_scope(span: Mapping[str, Any]) -> str | None:
    value = span.get("scope", span.get("instrumentationScope", span.get("instrumentation_scope")))
    if isinstance(value, Mapping) and isinstance(value.get("name"), str):
        return value["name"]
    return None


def spans_for_session(
    spans: Iterable[Mapping[str, Any]], session_id: str
) -> tuple[str, list[dict[str, Any]]]:
    """Find one exact trace through its session.id attribute."""

    materialized = [dict(span) for span in spans]
    trace_ids = {
        span_trace_id(span)
        for span in materialized
        if span_attributes(span).get("session.id") == session_id
    }
    trace_ids.discard(None)
    if len(trace_ids) != 1:
        raise ValueError(f"expected one trace for session {session_id!r}, found {len(trace_ids)}")
    trace_id = next(iter(trace_ids))
    selected = [span for span in materialized if span_trace_id(span) == trace_id]
    return trace_id, selected


def validate_observable_trace(
    spans: Iterable[Mapping[str, Any]], *, session_id: str, expected: Mapping[str, str]
) -> tuple[str, list[dict[str, Any]]]:
    trace_id, selected = spans_for_session(spans, session_id)
    supported = [span for span in selected if span_scope(span) == SUPPORTED_SCOPE]
    if not supported:
        raise ValueError("trace has no supported OpenInference LangChain scope")
    tool_spans = [
        span
        for span in supported
        if span_attributes(span).get("tool.name") == "get_invoice"
        or span.get("name") == "get_invoice"
    ]
    if not tool_spans:
        raise ValueError("trace has no get_invoice tool span")
    if not any(
        "disputed" in str(span_attributes(span).get("output.value", "")) for span in tool_spans
    ):
        raise ValueError("get_invoice trace evidence does not contain disputed status")
    root_attributes = [span_attributes(span) for span in selected]
    for key, value in expected.items():
        if not any(attributes.get(key) == value for attributes in root_attributes):
            raise ValueError(f"trace is missing expected provenance attribute {key}")
    return trace_id, selected
