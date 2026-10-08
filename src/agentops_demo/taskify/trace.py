"""Trace parsing and exact evidence-correlation helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.contracts.world_snapshot import BillingWorldSnapshot, snapshot_sha256
from agentops_demo.evaluation.dispute_policy import SUPPORTED_SCOPE, _successful, _tool_name
from agentops_demo.evaluation.trace_reader import span_attributes, span_scope, span_trace_id
from agentops_demo.taskify.contracts import FailureEvidence


class TraceCorrelationError(ValueError):
    pass


class WorldSnapshotError(ValueError):
    pass


def trace_sha256(trace: Mapping[str, Any]) -> str:
    encoded = json.dumps(trace, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def spans_from_trace(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = trace.get("evaluationInput", trace)
    if not isinstance(source, Mapping):
        raise TraceCorrelationError("trace must be an object")
    spans = source.get("sessionSpans", source.get("spans"))
    if not isinstance(spans, list) or not all(isinstance(span, Mapping) for span in spans):
        raise TraceCorrelationError("trace is missing session spans")
    return [dict(span) for span in spans]


def originating_config(attributes: Mapping[str, Any]) -> AgentConfig:
    try:
        return AgentConfig.model_validate(
            {
                "agent": {"source_revision": attributes["agentops.source_revision"]},
                "model": {
                    "provider": attributes["agentops.model_provider"],
                    "model_id": attributes["agentops.model_id"],
                },
                "prompt": {"version": attributes["agentops.prompt_version"]},
                "tools": {"version": attributes["agentops.tools_version"]},
                "harness": {
                    "framework": attributes["agentops.harness_framework"],
                    "version": attributes["agentops.harness_version"],
                },
            }
        )
    except (KeyError, ValueError) as exc:
        raise TraceCorrelationError("trace has incomplete candidate provenance") from exc


def correlate_trace(
    evidence: FailureEvidence, trace: Mapping[str, Any]
) -> tuple[Mapping[str, Any], list[dict[str, Any]], AgentConfig]:
    spans = spans_from_trace(trace)
    selected = [span for span in spans if span_trace_id(span) == evidence.source.trace_id]
    if not selected:
        raise TraceCorrelationError("trace does not contain the evidence trace ID")
    if any(span_trace_id(span) != evidence.source.trace_id for span in selected):
        raise TraceCorrelationError("trace ID mismatch")
    roots = [span_attributes(span) for span in selected if span.get("name") == "POST /invocations"]
    if len(roots) != 1:
        raise TraceCorrelationError("trace requires exactly one invocation root span")
    root = roots[0]
    if root.get("session.id") != evidence.source.session_id:
        raise TraceCorrelationError("session ID mismatch")
    config = originating_config(root)
    candidate = evidence.candidate
    expected = {
        "source_revision": candidate.source_revision,
        "model_provider": candidate.model_provider,
        "model_id": candidate.model_id,
        "prompt_version": candidate.prompt_version,
        "tools_version": candidate.tools_version,
        "harness_framework": candidate.harness_framework,
        "harness_version": candidate.harness_version,
    }
    actual_values = {
        "source_revision": config.agent.source_revision,
        "model_provider": config.model.provider,
        "model_id": config.model.model_id,
        "prompt_version": config.prompt.version,
        "tools_version": config.tools.version,
        "harness_framework": config.harness.framework,
        "harness_version": config.harness.version,
    }
    for field, expected_value in expected.items():
        actual = actual_values[field]
        if actual != expected_value:
            raise TraceCorrelationError(f"candidate {field} mismatch")
    if config.fingerprint() != candidate.agent_config_fingerprint:
        raise TraceCorrelationError("candidate fingerprint mismatch")
    instruction = root.get("agentops.invocation.instruction")
    if not isinstance(instruction, str) or not instruction:
        raise TraceCorrelationError("trace is missing invocation instruction")
    if instruction != evidence.source.instruction:
        raise TraceCorrelationError("instruction mismatch")
    if not any(span_scope(span) == SUPPORTED_SCOPE for span in selected):
        raise TraceCorrelationError("trace has no supported OpenInference scope")
    return root, selected, config


def snapshot_from_root(root: Mapping[str, Any]) -> BillingWorldSnapshot:
    serialized = root.get("agentops.world_state.before")
    if not isinstance(serialized, str) or not serialized:
        raise WorldSnapshotError(
            "trace is missing pre-invocation world snapshot; enable "
            "synthetic_demo_trace_content_capture for synthetic billing data "
            "and collect a new trace"
        )
    try:
        snapshot = BillingWorldSnapshot.model_validate_json(serialized)
    except ValueError as exc:
        raise WorldSnapshotError("world snapshot is invalid") from exc
    if root.get("agentops.world_snapshot.schema_version") != snapshot.schema_version:
        raise WorldSnapshotError("world snapshot schema version mismatch")
    if root.get("agentops.world_snapshot.kind") != "billing-demo":
        raise WorldSnapshotError("world snapshot kind is unsupported")
    if root.get("agentops.world_snapshot.sha256") != snapshot_sha256(snapshot):
        raise WorldSnapshotError("world snapshot digest mismatch")
    return snapshot


def ordered_successful_tool_spans(
    spans: list[dict[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]]:
    result: list[tuple[str, Mapping[str, Any], int, str]] = []
    for span in spans:
        attributes = span_attributes(span)
        tool = _tool_name(span, attributes)
        if tool is None:
            raw_tool = attributes.get("tool.name")
            if (
                attributes.get("openinference.span.kind") == "TOOL"
                and isinstance(raw_tool, str)
                and raw_tool
            ):
                tool = raw_tool
            else:
                continue
        if span_scope(span) != SUPPORTED_SCOPE:
            raise TraceCorrelationError("tool span has unsupported scope")
        if "input.value" not in attributes or "output.value" not in attributes:
            raise TraceCorrelationError(f"tool span {tool!r} lacks input or output")
        if not _successful(span, attributes["output.value"]):
            continue
        raw_start = span.get("startTimeUnixNano", span.get("start_time_unix_nano"))
        if isinstance(raw_start, bool):
            raise TraceCorrelationError(f"tool span {tool!r} has invalid start timestamp")
        try:
            start = int(raw_start)
        except (TypeError, ValueError) as exc:
            raise TraceCorrelationError(f"tool span {tool!r} has invalid start timestamp") from exc
        if start < 0:
            raise TraceCorrelationError(f"tool span {tool!r} has negative start timestamp")
        span_id = span.get("spanId")
        if not isinstance(span_id, str) or not span_id:
            raise TraceCorrelationError(f"tool span {tool!r} has invalid span ID")
        result.append((tool, attributes, start, span_id))
    result.sort(key=lambda item: (item[2], item[3]))
    return [(tool, attributes) for tool, attributes, _, _ in result]
