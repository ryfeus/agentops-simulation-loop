"""Materialize one trace-backed interactive AgentCore failure for taskification."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from agentops_demo.evaluation.trace_reader import span_attributes, span_trace_id
from agentops_demo.taskify.contracts import FailureEvidence
from agentops_demo.taskify.trace import (
    TraceCorrelationError,
    ordered_successful_tool_spans,
    originating_config,
    spans_from_trace,
)
from scripts.dsql_admin import terraform_outputs
from scripts.evaluation_deployment import EVALUATION_MANIFEST, _load_object
from scripts.read_online_evaluation_results import wait_for_result
from scripts.taskify import fetch_trace

DEFAULT_OUTPUT = Path(".agentcore/evaluation/failure.json")


def build_failure_evidence(
    *,
    evaluation: Mapping[str, Any],
    trace: Mapping[str, Any],
    session_id: str,
    trace_id: str,
    service_name: str,
    trace_log_group: str,
    result_log_group: str,
    runtime_arn: str | None = None,
) -> dict[str, Any]:
    """Create strict failure evidence only from one correlated FAIL result and trace."""

    if evaluation.get("label") != "FAIL" or evaluation.get("value") != 0.0:
        raise ValueError("online evaluation is not a FAIL result with score 0.0")
    if evaluation.get("session_id") != session_id or evaluation.get("trace_id") != trace_id:
        raise ValueError("online evaluation does not match the requested session and trace")
    spans = [span for span in spans_from_trace(trace) if span_trace_id(span) == trace_id]
    if not spans:
        raise TraceCorrelationError("trace does not contain the requested trace ID")
    roots = [span_attributes(span) for span in spans if span.get("name") == "POST /invocations"]
    if len(roots) != 1:
        raise TraceCorrelationError("trace requires exactly one invocation root span")
    root = roots[0]
    if root.get("session.id") != session_id:
        raise TraceCorrelationError("trace session ID does not match the requested session")
    instruction = root.get("agentops.invocation.instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise TraceCorrelationError("trace is missing invocation instruction")
    config = originating_config(root)
    trajectory = [tool for tool, _attributes in ordered_successful_tool_spans(spans)]
    evidence = FailureEvidence.model_validate(
        {
            "schema_version": "1",
            "evaluator": {
                "id": evaluation.get("evaluator_id"),
                "name": evaluation.get("evaluator"),
                "label": evaluation.get("label"),
                "value": evaluation.get("value"),
                "explanation": evaluation.get("explanation"),
            },
            "source": {
                "runtime_arn": runtime_arn,
                "service_name": service_name,
                "trace_log_group": trace_log_group,
                "session_id": session_id,
                "trace_id": trace_id,
                "result_log_group": result_log_group,
                "result_timestamp": evaluation.get("timestamp"),
                "instruction": instruction,
            },
            "candidate": {
                "source_revision": config.agent.source_revision,
                "agent_config_fingerprint": config.fingerprint(),
                "model_provider": config.model.provider,
                "model_id": config.model.model_id,
                "prompt_version": config.prompt.version,
                "tools_version": config.tools.version,
                "harness_framework": config.harness.framework,
                "harness_version": config.harness.version,
            },
            "trajectory": trajectory,
            "trace_span_count": len(spans),
        }
    )
    return evidence.model_dump(mode="json")


def capture(
    *,
    session_id: str,
    trace_id: str,
    timeout: float,
    lookback_minutes: float,
    evaluation_reader: Callable[..., dict[str, Any]] = wait_for_result,
    trace_fetcher: Callable[..., dict[str, Any]] = fetch_trace,
) -> dict[str, Any]:
    """Fetch correlated production evidence without re-running the evaluator."""

    if not session_id.strip() or not trace_id.strip():
        raise ValueError("session_id and trace_id must not be blank")
    if timeout <= 0 or lookback_minutes <= 0:
        raise ValueError("timeout and lookback_minutes must be positive")
    manifest = _load_object(EVALUATION_MANIFEST, "evaluation manifest")
    if manifest.get("state") != "deployed":
        raise RuntimeError("online evaluation is not deployed")
    outputs = terraform_outputs()
    result_log_group = str(manifest["online_evaluation"]["result_log_group"])
    evaluation = evaluation_reader(
        log_group=result_log_group,
        evaluator_id=str(manifest["evaluator"]["id"]),
        online_evaluation_config_id=str(manifest["online_evaluation"]["id"]),
        session_id=session_id,
        trace_id=trace_id,
        start_time=time.time() - lookback_minutes * 60,
        timeout=timeout,
    )
    trace_log_group = str(outputs["agentcore_trace_log_group"])
    trace = trace_fetcher(log_group=trace_log_group, trace_id=trace_id)
    runtime_arn = outputs.get("agentcore_runtime_arn")
    return build_failure_evidence(
        evaluation=evaluation,
        trace=trace,
        session_id=session_id,
        trace_id=trace_id,
        service_name=str(outputs["agentcore_runtime_service_name"]),
        trace_log_group=trace_log_group,
        result_log_group=result_log_group,
        runtime_arn=runtime_arn if isinstance(runtime_arn, str) and runtime_arn else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--lookback-minutes", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    evidence = capture(
        session_id=args.session_id,
        trace_id=args.trace_id,
        timeout=args.timeout,
        lookback_minutes=args.lookback_minutes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
