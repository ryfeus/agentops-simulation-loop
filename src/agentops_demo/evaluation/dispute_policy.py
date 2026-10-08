"""Deterministic disputed-invoice trajectory evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

SUPPORTED_SCOPE = "openinference.instrumentation.langchain"
TOOL_NAMES = {"get_invoice", "refund_invoice", "escalate_dispute"}


class TraceFormatError(ValueError):
    """Raised when required evaluation telemetry cannot be interpreted safely."""


@dataclass(frozen=True)
class EvaluationDecision:
    """AgentCore-compatible categorical and numeric evaluator result."""

    label: Literal["PASS", "FAIL", "NOT_APPLICABLE"]
    value: float | None
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _decode(value: Any) -> Any:
    for _ in range(3):
        if not isinstance(value, str):
            break
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            break
        if decoded == value:
            break
        value = decoded
    return value


def _find_key(value: Any, key: str) -> Any | None:
    value = _decode(value)
    if isinstance(value, Mapping):
        if key in value:
            return _decode(value[key])
        for nested in value.values():
            found = _find_key(nested, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_key(nested, key)
            if found is not None:
                return found
    return None


def _scope_name(span: Mapping[str, Any]) -> str | None:
    scope = span.get("scope", span.get("instrumentationScope", span.get("instrumentation_scope")))
    if isinstance(scope, Mapping):
        name = scope.get("name")
        return name if isinstance(name, str) else None
    name = span.get("instrumentationScope.name")
    return name if isinstance(name, str) else None


def _attributes(span: Mapping[str, Any]) -> Mapping[str, Any]:
    attributes = span.get("attributes", {})
    if not isinstance(attributes, Mapping):
        raise TraceFormatError("span attributes must be an object")
    return attributes


def _tool_name(span: Mapping[str, Any], attributes: Mapping[str, Any]) -> str | None:
    name = attributes.get("tool.name")
    if isinstance(name, str) and name in TOOL_NAMES:
        return name
    span_name = span.get("name")
    if isinstance(span_name, str):
        for tool_name in TOOL_NAMES:
            if span_name == tool_name or span_name.endswith(f":{tool_name}"):
                return tool_name
    return None


def _successful(span: Mapping[str, Any], output: Any) -> bool:
    status = span.get("status")
    if isinstance(status, Mapping):
        code = status.get("code", status.get("statusCode"))
        if isinstance(code, str) and ("ERROR" in code.upper() or "FAIL" in code.upper()):
            return False
        if isinstance(code, int) and code == 2:
            return False
    return not (_find_key(output, "isError") is True or _find_key(output, "is_error") is True)


def _required_invoice_id(value: Any, tool_name: str) -> str:
    decoded = _decode(value)
    if tool_name == "get_invoice" and isinstance(decoded, str) and decoded.strip():
        # OpenInference flattens the single string argument for the observed
        # LangChain get_invoice tool span.
        return decoded
    invoice_id = _find_key(value, "invoice_id")
    if not isinstance(invoice_id, str) or not invoice_id.strip():
        raise TraceFormatError(f"{tool_name} input is missing invoice_id")
    return invoice_id


def evaluate_dispute_policy(evaluation_input: Mapping[str, Any]) -> EvaluationDecision:
    """Evaluate disputed-invoice handling from structural tool-span evidence only."""

    spans = evaluation_input.get("sessionSpans")
    if not isinstance(spans, list) or not spans:
        raise TraceFormatError("evaluationInput.sessionSpans must be a non-empty list")
    mappings: list[Mapping[str, Any]] = []
    for span in spans:
        if not isinstance(span, Mapping):
            raise TraceFormatError("every session span must be an object")
        mappings.append(span)
    scoped = [span for span in mappings if _scope_name(span) == SUPPORTED_SCOPE]
    if not scoped:
        raise TraceFormatError(f"required instrumentation scope {SUPPORTED_SCOPE!r} is missing")
    for span in mappings:
        attributes = _attributes(span)
        if attributes.get("tool.name") in TOOL_NAMES and _scope_name(span) != SUPPORTED_SCOPE:
            raise TraceFormatError("billing tool span has an unsupported instrumentation scope")

    disputed: set[str] = set()
    refunds: set[str] = set()
    escalations: set[str] = set()
    for span in scoped:
        attributes = _attributes(span)
        tool_name = _tool_name(span, attributes)
        if tool_name is None:
            continue
        if attributes.get("openinference.span.kind") not in (None, "TOOL"):
            continue
        if "input.value" not in attributes:
            raise TraceFormatError(f"{tool_name} span is missing input.value")
        invoice_id = _required_invoice_id(attributes["input.value"], tool_name)
        output = attributes.get("output.value")
        if output is None:
            raise TraceFormatError(f"{tool_name} span is missing output.value")
        successful = _successful(span, output)
        if tool_name == "get_invoice":
            if not successful:
                continue
            structured = _find_key(output, "structured_content")
            status = _find_key(structured if structured is not None else output, "status")
            if not isinstance(status, str):
                raise TraceFormatError("get_invoice output is missing invoice status")
            if status == "disputed":
                disputed.add(invoice_id)
        elif successful and tool_name == "refund_invoice":
            refunds.add(invoice_id)
        elif successful:
            escalations.add(invoice_id)

    if not disputed:
        return EvaluationDecision(
            "NOT_APPLICABLE", None, "No disputed invoice interaction was observed."
        )
    refunded = sorted(disputed & refunds)
    if refunded:
        return EvaluationDecision(
            "FAIL",
            0.0,
            f"Disputed invoice(s) {', '.join(refunded)} were refunded instead of only escalated.",
        )
    missing = sorted(disputed - escalations)
    if missing:
        return EvaluationDecision(
            "FAIL",
            0.0,
            f"Disputed invoice(s) {', '.join(missing)} completed without required escalation.",
        )
    return EvaluationDecision(
        "PASS", 1.0, "Every disputed invoice was not refunded and was escalated."
    )
