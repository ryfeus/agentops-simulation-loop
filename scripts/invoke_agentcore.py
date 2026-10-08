"""Invoke the deployed AgentCore runtime directly through Boto3."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from agentops_demo.agentcore.candidates import SUPPORTED_CANDIDATES
from agentops_demo.agentcore.request import CandidateSelector
from scripts.dsql_admin import terraform_outputs

DEFAULT_INSTRUCTION = (
    "Inspect invoice inv-123 using the billing tools and report its current status."
)
DEFAULT_OUTPUT = Path(".agentcore/smoke/result.json")


def invoke(
    instruction: str,
    *,
    session_id: str | None = None,
    candidate: CandidateSelector | None = None,
    timeout_seconds: float = 120.0,
) -> tuple[dict[str, Any], str]:
    """Invoke AgentCore with an explicit bounded client read timeout."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if session_id is not None and not session_id.strip():
        raise ValueError("session_id must not be blank")
    from botocore.config import Config

    from scripts.aws_context import verified_session

    session = verified_session()
    outputs = terraform_outputs()
    region = str(outputs["aws_region"])
    runtime_arn = str(outputs["agentcore_runtime_arn"])
    if region != "us-west-2" or not runtime_arn:
        raise RuntimeError("a us-west-2 AgentCore runtime must be deployed")
    requested_session_id = session_id or str(uuid.uuid4())
    client = session.client(
        "bedrock-agentcore",
        region_name=region,
        config=Config(
            connect_timeout=min(10.0, timeout_seconds),
            read_timeout=timeout_seconds,
            retries={"max_attempts": 1, "mode": "standard"},
        ),
    )
    response = client.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        runtimeSessionId=requested_session_id,
        qualifier="DEFAULT",
        contentType="application/json",
        accept="application/json",
        payload=json.dumps(
            {
                "instruction": instruction,
                **({"candidate": candidate} if candidate is not None else {}),
            }
        ).encode(),
    )
    body = response["response"].read()
    result = json.loads(body.decode())
    if not isinstance(result, dict):
        raise RuntimeError("AgentCore returned a non-object response")
    returned_session_id = response.get("runtimeSessionId")
    if not isinstance(returned_session_id, str) or returned_session_id != requested_session_id:
        raise RuntimeError(
            "AgentCore returned an unexpected runtime session ID: "
            f"expected {requested_session_id!r}, received {returned_session_id!r}"
        )
    return result, returned_session_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--candidate", choices=SUPPORTED_CANDIDATES)
    args = parser.parse_args(argv)
    result, session_id = invoke(
        args.instruction,
        candidate=args.candidate,
        timeout_seconds=args.timeout,
    )
    payload = {"runtime_session_id": session_id, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
