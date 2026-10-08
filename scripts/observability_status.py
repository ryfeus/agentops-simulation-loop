"""Print the read-only AgentCore runtime observability identity."""

from __future__ import annotations

import json

from scripts.dsql_admin import terraform_outputs


def main() -> int:
    outputs = terraform_outputs()
    payload = {
        "runtime_arn": outputs.get("agentcore_runtime_arn"),
        "service_name": outputs.get("agentcore_runtime_service_name"),
        "trace_log_group": outputs.get("agentcore_trace_log_group"),
        "trace_stream": "spans",
    }
    if not all(payload[key] for key in ("runtime_arn", "service_name", "trace_log_group")):
        raise RuntimeError("Terraform does not identify a deployed AgentCore runtime")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
