"""AWS Lambda adapter for the deterministic dispute-policy evaluator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agentops_demo.evaluation.dispute_policy import TraceFormatError, evaluate_dispute_policy

EVALUATOR_NAME = "DisputePolicyCompliance"


def _invalid(message: str) -> dict[str, str]:
    return {"errorCode": "INVALID_EVALUATION_REQUEST", "errorMessage": message}


def _valid_identity(event: Mapping[str, Any], evaluator_name: str) -> bool:
    """Accept the documented identity or the identity-less envelope AWS emits today."""

    evaluator_id = event.get("evaluatorId")
    received_name = event.get("evaluatorName")
    if evaluator_id is None and received_name is None:
        # AgentCore's live Evaluate and online-evaluation paths omit both fields
        # before Lambda invocation, then attach them to the outer result.
        return True
    return (
        received_name == evaluator_name
        and isinstance(evaluator_id, str)
        and evaluator_id.startswith(f"{evaluator_name}-")
        and len(evaluator_id.removeprefix(f"{evaluator_name}-")) == 10
    )


def evaluate_handler(
    event: Any,
    *,
    evaluator_name: str,
    evaluator: Any,
) -> dict[str, object]:
    """Validate a shared AgentCore envelope then run one deterministic evaluator."""

    if not isinstance(event, Mapping):
        return _invalid("Evaluation event must be an object.")
    if event.get("schemaVersion") != "1.0":
        return _invalid("schemaVersion must be '1.0'.")
    if not _valid_identity(event, evaluator_name):
        return _invalid("Evaluator identity is invalid or incomplete.")
    evaluator_id = event.get("evaluatorId") or evaluator_name
    if event.get("evaluationLevel") != "TRACE":
        return _invalid("evaluationLevel must be 'TRACE'.")
    evaluation_input = event.get("evaluationInput")
    target = event.get("evaluationTarget")
    if not isinstance(evaluation_input, Mapping):
        return _invalid("evaluationInput must be an object.")
    if not isinstance(target, Mapping):
        return _invalid("evaluationTarget must be an object.")
    trace_ids = target.get("traceIds")
    if (
        not isinstance(trace_ids, list)
        or not trace_ids
        or not all(isinstance(item, str) and item for item in trace_ids)
    ):
        return _invalid("evaluationTarget.traceIds must be a non-empty string list.")
    spans = evaluation_input.get("sessionSpans")
    if not isinstance(spans, list):
        return _invalid("evaluationInput.sessionSpans must be a list.")
    filtered = [
        span for span in spans if isinstance(span, Mapping) and span.get("traceId") in trace_ids
    ]
    if not filtered:
        return _invalid("No session span matches evaluationTarget.traceIds.")
    try:
        decision = evaluator({"sessionSpans": filtered})
    except TraceFormatError as exc:
        print(json.dumps({"event": "evaluation_error", "error_code": "TRACE_FORMAT_UNSUPPORTED"}))
        return {"errorCode": "TRACE_FORMAT_UNSUPPORTED", "errorMessage": str(exc)}
    print(
        json.dumps(
            {
                "event": "evaluation_completed",
                "evaluator_id": evaluator_id,
                "label": decision.label,
                "trace_ids": trace_ids,
            },
            sort_keys=True,
        )
    )
    return decision.to_dict()


def handler(event: Any, _context: Any) -> dict[str, object]:
    """Legacy DisputePolicyCompliance Lambda entry point."""

    return evaluate_handler(
        event,
        evaluator_name=EVALUATOR_NAME,
        evaluator=evaluate_dispute_policy,
    )
