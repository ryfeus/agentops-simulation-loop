"""Minimal interactive chat client for an opt-in Bedrock-backed local agent."""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid

from agentops_demo.agent.local import (
    local_agent_config_from_environment,
    local_candidate_from_environment,
)
from agentops_demo.agent.runtime import create_session_runtime
from agentops_demo.agentcore.candidates import SUPPORTED_CANDIDATES


async def run_chat(*, candidate: str | None = None) -> None:
    selected = local_candidate_from_environment(candidate)
    config = local_agent_config_from_environment(selected)
    runtime = await create_session_runtime(
        config=config,
        mcp_url=os.getenv("BILLING_MCP_URL", "http://127.0.0.1:8000/mcp"),
    )
    session_id = str(uuid.uuid4())
    print(f"Billing agent ready ({selected}). Enter 'quit' to exit.")
    while True:
        try:
            instruction = input("you> ").strip()
        except EOFError:
            break
        if instruction.lower() in {"quit", "exit"}:
            break
        if not instruction:
            continue
        result = await runtime.invoke(session_id=session_id, instruction=instruction)
        for tool in result["completed_tools"]:
            print(f"tool[{tool}]> completed")
        print(f"agent> {result['final_response']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=SUPPORTED_CANDIDATES)
    args = parser.parse_args(argv)
    asyncio.run(run_chat(candidate=args.candidate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
