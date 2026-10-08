"""Run one configured billing-agent invocation and emit diagnostics."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from agentops_demo.agent.graph import build_agent as _build_agent
from agentops_demo.agent.model import create_model as _create_model
from agentops_demo.agent.runtime import AgentSessionRuntime
from agentops_demo.agent.runtime import create_session_runtime as _create_session_runtime
from agentops_demo.agent.runtime import invoke_once as _invoke_once
from agentops_demo.contracts.agent_config import AgentConfig

create_model = _create_model
build_agent = _build_agent


async def create_session_runtime(*, config: AgentConfig, mcp_url: str) -> AgentSessionRuntime:
    """Compatibility wrapper around the reusable agent runtime factory."""

    return await _create_session_runtime(
        config=config,
        mcp_url=mcp_url,
        model_factory=create_model,
        graph_factory=build_agent,
    )


async def invoke_once(*, config: AgentConfig, instruction: str, mcp_url: str) -> dict[str, Any]:
    """Compatibility wrapper around the reusable one-shot invocation helper."""

    return await _invoke_once(
        config=config,
        instruction=instruction,
        mcp_url=mcp_url,
        model_factory=create_model,
        graph_factory=build_agent,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one configured billing-agent invocation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    config = AgentConfig.model_validate_json(args.config.read_text())
    instruction = args.instruction_file.read_text().strip()
    if not instruction:
        parser.error("instruction file must not be blank")
    result = asyncio.run(invoke_once(config=config, instruction=instruction, mcp_url=args.mcp_url))
    serialized = json.dumps(result, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{serialized}\n")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
