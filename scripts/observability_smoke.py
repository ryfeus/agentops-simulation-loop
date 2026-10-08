"""Invoke AgentCore and require an evaluation-compatible correlated trace."""

from __future__ import annotations

import argparse
import json

from scripts.find_agentcore_trace import DEFAULT_EVIDENCE, DEFAULT_OUTPUT, find_trace
from scripts.invoke_agentcore import invoke

INSTRUCTION = "Inspect invoice inv-123 using the billing tools and report its current status."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    result, session_id = invoke(INSTRUCTION)
    if "get_invoice" not in result.get("completed_tools", []):
        raise RuntimeError("AgentCore invocation did not complete get_invoice")
    envelope, evidence = find_trace(session_id=session_id, timeout=args.timeout)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    DEFAULT_EVIDENCE.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
