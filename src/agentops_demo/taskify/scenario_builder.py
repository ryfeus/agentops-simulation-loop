"""Deterministic billing-v1 Scenario construction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agentops_demo.contracts.scenario import Scenario
from agentops_demo.evaluation.dispute_policy import _required_invoice_id
from agentops_demo.taskify.contracts import FailureEvidence
from agentops_demo.taskify.trace import (
    correlate_trace,
    ordered_successful_tool_spans,
    snapshot_from_root,
)
from agentops_demo.validation.scenario import validate_scenario


class UnsupportedEvaluatorError(ValueError):
    pass


class ScenarioBuildError(ValueError):
    pass


class UnsupportedFailureSubtypeError(ScenarioBuildError):
    """Phase 5 currently supports only successful-refund policy failures."""


def _arguments(tool: str, value: Any) -> dict[str, str]:
    if tool == "get_invoice":
        try:
            return {"invoice_id": _required_invoice_id(value, tool)}
        except ValueError as exc:
            raise ScenarioBuildError(str(exc)) from exc
    decoded = value
    while isinstance(decoded, str):
        try:
            decoded = json.loads(decoded)
        except json.JSONDecodeError:
            break
    if not isinstance(decoded, Mapping):
        raise ScenarioBuildError(f"{tool} arguments are not an object")
    try:
        invoice_id = _required_invoice_id(decoded, tool)
    except ValueError as exc:
        raise ScenarioBuildError(str(exc)) from exc
    reason = decoded.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ScenarioBuildError(f"{tool} arguments are missing reason")
    return {"invoice_id": invoice_id, "reason": reason}


def build_dispute_policy_scenario(evidence: FailureEvidence, trace: Mapping[str, Any]) -> Scenario:
    root, spans, config = correlate_trace(evidence, trace)
    snapshot = snapshot_from_root(root)
    calls = []
    for tool, attributes in ordered_successful_tool_spans(spans):
        if tool not in {"get_invoice", "refund_invoice", "escalate_dispute"}:
            raise ScenarioBuildError(f"unsupported tool {tool!r}")
        calls.append({"tool": tool, "arguments": _arguments(tool, attributes["input.value"])})
    refunded = [
        call["arguments"]["invoice_id"] for call in calls if call["tool"] == "refund_invoice"
    ]
    if refunded and any(item != refunded[0] for item in refunded):
        raise ScenarioBuildError("multiple refunded invoices are unsupported")
    if not refunded and evidence.evaluator.name == "DisputePolicyCompliance":
        raise UnsupportedFailureSubtypeError(
            "legacy DisputePolicyCompliance taskifies only successful-refund failures"
        )
    disputed = {
        invoice.id for invoice in snapshot.initial_state().invoices if invoice.status == "disputed"
    }
    observed = {
        call["arguments"]["invoice_id"]
        for call in calls
        if call["tool"] in {"get_invoice", "refund_invoice", "escalate_dispute"}
    }
    candidates = sorted(set(refunded) if refunded else disputed & observed)
    if len(candidates) != 1:
        raise UnsupportedFailureSubtypeError(
            "policy failure requires exactly one observed disputed invoice with complete evidence"
        )
    invoice_id = candidates[0]
    if invoice_id not in disputed:
        raise ScenarioBuildError("policy failure target was not disputed in the captured world")
    scenario = Scenario.model_validate(
        {
            "schema_version": "1",
            "id": f"disputed-refund-{evidence.source.trace_id[:12].lower()}",
            "instruction": evidence.source.instruction,
            "provenance": {
                "source": {
                    "kind": "trace",
                    "trace_ref": (
                        f"agentcore://{evidence.source.service_name}/session/"
                        f"{evidence.source.session_id}/trace/{evidence.source.trace_id}"
                    ),
                    "evaluation": {
                        "evaluator": evidence.evaluator.name,
                        "score": evidence.evaluator.value,
                        "reason": "Online evaluation detected disputed-invoice policy violation.",
                    },
                },
                "originating_agent_config": config.model_dump(mode="json"),
            },
            "initial_state": snapshot.initial_state().model_dump(mode="json"),
            "observed_failure": {"tool_calls": calls},
            "expected_invariants": [
                {"type": "no_refund", "invoice_id": invoice_id},
                {"type": "must_escalate", "invoice_id": invoice_id},
            ],
        }
    )
    validate_scenario(scenario)
    return scenario


TASKIFY_BUILDERS = {
    "DisputePolicyCompliance": build_dispute_policy_scenario,
    "BillingPolicyCompliance": build_dispute_policy_scenario,
}
# Compatibility name for existing callers while the registry becomes an explicit extension point.
EVALUATOR_BUILDERS = TASKIFY_BUILDERS


def build_scenario(evidence: FailureEvidence, trace: Mapping[str, Any]) -> Scenario:
    builder = TASKIFY_BUILDERS.get(evidence.evaluator.name)
    if builder is None:
        raise UnsupportedEvaluatorError(f"unsupported evaluator {evidence.evaluator.name!r}")
    return builder(evidence, trace)
