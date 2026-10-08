"""Locate and validate an AgentCore LangGraph trace in its unified CloudWatch stream."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from agentops_demo.agentcore.candidates import SUPPORTED_CANDIDATES, resolve_candidate
from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.evaluation.trace_reader import parse_json_record, validate_observable_trace
from scripts.agentcore_package import load_config
from scripts.dsql_admin import terraform_outputs

DEFAULT_OUTPUT = Path(".agentcore/observability/trace.json")
DEFAULT_EVIDENCE = Path(".agentcore/observability/evidence.json")


def _events(client: Any, log_group: str, start_ms: int) -> list[dict[str, Any]]:
    response = client.filter_log_events(
        logGroupName=log_group,
        logStreamNames=["spans"],
        startTime=start_ms,
    )
    records: list[dict[str, Any]] = []
    for event in response.get("events", []):
        try:
            records.append(parse_json_record(event["message"]))
        except (KeyError, ValueError):
            continue
    return records


def find_trace(
    *,
    session_id: str,
    config: AgentConfig | None = None,
    timeout: float = 600.0,
    started_at: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from scripts.aws_context import verified_session

    outputs = terraform_outputs()
    log_group = str(outputs["agentcore_trace_log_group"])
    service_name = str(outputs["agentcore_runtime_service_name"])
    config = config or load_config()
    expected = {
        "agentops.source_revision": config.agent.source_revision,
        "agentops.agent_config_fingerprint": config.fingerprint(),
        "agentops.model_provider": config.model.provider,
        "agentops.model_id": config.model.model_id,
        "agentops.prompt_version": config.prompt.version,
        "agentops.tools_version": config.tools.version,
        "agentops.harness_framework": config.harness.framework,
        "agentops.harness_version": config.harness.version,
    }
    client = verified_session().client("logs", region_name="us-west-2")
    start_ms = int(((started_at or time.time()) - 300) * 1000)
    deadline = time.monotonic() + timeout
    last_error = "no spans received"
    while time.monotonic() < deadline:
        try:
            trace_id, spans = validate_observable_trace(
                _events(client, log_group, start_ms), session_id=session_id, expected=expected
            )
            envelope = {
                "schemaVersion": "1.0",
                "evaluationInput": {"sessionSpans": spans},
                "evaluationTarget": {"traceIds": [trace_id], "spanIds": []},
            }
            evidence = {
                "schema_version": "1",
                "session_id": session_id,
                "trace_id": trace_id,
                "source_revision": config.agent.source_revision,
                "agent_config_fingerprint": config.fingerprint(),
                "service_name": service_name,
                "trace_log_group": log_group,
                "instrumentation_scope": "openinference.instrumentation.langchain",
                "required_tool": "get_invoice",
            }
            return envelope, evidence
        except ValueError as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(
        f"trace discovery timed out for session={session_id} service={service_name} "
        f"log_group={log_group}: {last_error}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--candidate", choices=SUPPORTED_CANDIDATES)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args(argv)
    config = resolve_candidate(load_config(), args.candidate) if args.candidate else None
    envelope, evidence = find_trace(session_id=args.session_id, config=config, timeout=args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    args.evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
