"""Create a deterministic, credential-free fixture from a captured AgentCore trace."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentops_demo.evaluation.dispute_policy import SUPPORTED_SCOPE

DEFAULT_INPUT = Path(".agentcore/observability/compliant-trace.json")
DEFAULT_OUTPUT = Path("tests/fixtures/agentcore/langgraph_billing_trace.json")
TRACE_ID = "00000000000000000000000000000001"
SESSION_ID = "session-sanitized"
SOURCE_REVISION = "b" * 40
CONFIG_FINGERPRINT = "a" * 64
ROOT_ATTRIBUTES = {
    "agentops.agent_config_fingerprint",
    "agentops.billing_policy_mode",
    "agentops.harness_framework",
    "agentops.harness_version",
    "agentops.model_id",
    "agentops.model_provider",
    "agentops.prompt_version",
    "agentops.source_revision",
    "agentops.tools_version",
}
MODEL_ATTRIBUTES = {
    "gen_ai.provider.name",
    "gen_ai.request.model",
    "llm.model_name",
    "llm.provider",
}


def _scope(span: Mapping[str, Any]) -> Mapping[str, Any]:
    value = span.get("scope")
    return value if isinstance(value, Mapping) else {}


def _decode(value: Any) -> Any:
    for _ in range(4):
        if not isinstance(value, str):
            return value
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if decoded == value:
            return value
        value = decoded
    return value


def _find_mapping(value: Any, key: str) -> Mapping[str, Any] | None:
    value = _decode(value)
    if isinstance(value, Mapping):
        candidate = value.get(key)
        if isinstance(candidate, Mapping):
            return candidate
        for nested in value.values():
            found = _find_mapping(nested, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_mapping(nested, key)
            if found is not None:
                return found
    return None


def _canonical_tool_output(value: Any) -> str:
    structured = _find_mapping(value, "structured_content")
    if structured is None:
        raise ValueError("tool output does not contain structured_content")
    result = structured.get("result", structured)
    if not isinstance(result, Mapping):
        raise ValueError("tool structured_content result must be an object")
    return json.dumps(dict(result), separators=(",", ":"), sort_keys=True)


def _sanitize_attributes(span: Mapping[str, Any]) -> dict[str, Any]:
    raw = span.get("attributes")
    if not isinstance(raw, Mapping):
        raise ValueError("every retained span must have object attributes")
    attributes: dict[str, Any] = {
        "PlatformType": raw.get("PlatformType", "AWS::BedrockAgentCore"),
        "aws.local.service": "agentops_demo_dev.DEFAULT",
        "session.id": SESSION_ID,
    }
    if span.get("name") == "POST /invocations":
        for key in sorted(ROOT_ATTRIBUTES):
            if key not in raw:
                raise ValueError(f"invocation span is missing {key}")
            attributes[key] = raw[key]
        attributes["agentops.source_revision"] = SOURCE_REVISION
        attributes["agentops.agent_config_fingerprint"] = CONFIG_FINGERPRINT
        return attributes

    span_kind = raw.get("openinference.span.kind")
    if not isinstance(span_kind, str):
        raise ValueError(f"OpenInference span {span.get('name')!r} is missing its kind")
    attributes["openinference.span.kind"] = span_kind
    for key in sorted(MODEL_ATTRIBUTES):
        if key in raw:
            attributes[key] = raw[key]
    tool_name = raw.get("tool.name")
    if isinstance(tool_name, str):
        for key in ("input.value", "output.value"):
            if key not in raw:
                raise ValueError(f"tool span {tool_name!r} is missing {key}")
        attributes.update(
            {
                "input.value": raw["input.value"],
                "openinference.span.kind": "TOOL",
                "output.mime_type": "application/json",
                "output.value": _canonical_tool_output(raw["output.value"]),
                "tool.name": tool_name,
            }
        )
    return attributes


def sanitize_trace(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Retain verified semantic spans while removing request/authentication metadata."""

    evaluation_input = envelope.get("evaluationInput")
    if not isinstance(evaluation_input, Mapping):
        raise ValueError("trace is missing evaluationInput")
    spans = evaluation_input.get("sessionSpans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("trace is missing evaluationInput.sessionSpans")
    retained = [
        span
        for span in spans
        if isinstance(span, Mapping)
        and (
            _scope(span).get("name") == SUPPORTED_SCOPE
            or (
                span.get("name") == "POST /invocations"
                and isinstance(span.get("attributes"), Mapping)
                and "agentops.source_revision" in span["attributes"]
            )
        )
    ]
    if not retained:
        raise ValueError("trace has no evaluator-relevant spans")
    original_ids = [span.get("spanId") for span in retained]
    if any(not isinstance(span_id, str) or not span_id for span_id in original_ids):
        raise ValueError("every retained span must have a spanId")
    id_map = {span_id: f"{index:016x}" for index, span_id in enumerate(original_ids, start=1)}
    sanitized: list[dict[str, Any]] = []
    for span in retained:
        scope = _scope(span)
        sanitized_span: dict[str, Any] = {
            "attributes": _sanitize_attributes(span),
            "name": span["name"],
            "scope": {
                "name": scope.get("name"),
                **({"version": scope["version"]} if "version" in scope else {}),
            },
            "spanId": id_map[span["spanId"]],
            "traceId": TRACE_ID,
        }
        parent_id = span.get("parentSpanId")
        if parent_id in id_map:
            sanitized_span["parentSpanId"] = id_map[parent_id]
        for key in ("kind", "status"):
            if key in span:
                sanitized_span[key] = span[key]
        sanitized.append(sanitized_span)
    return {
        "evaluationInput": {"sessionSpans": sanitized},
        "evaluationLevel": "TRACE",
        "evaluationReferenceInputs": [],
        "evaluationTarget": {"spanIds": [], "traceIds": [TRACE_ID]},
        "evaluatorId": "DisputePolicyCompliance-ABCDEFGHIJ",
        "evaluatorName": "DisputePolicyCompliance",
        "schemaVersion": "1.0",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    raw = json.loads(args.input.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError("captured trace must be a JSON object")
    result = sanitize_trace(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(result['evaluationInput']['sessionSpans'])} sanitized spans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
