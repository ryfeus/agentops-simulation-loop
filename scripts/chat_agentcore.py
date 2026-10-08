"""Run a small interactive chat against the deployed AgentCore runtime."""

from __future__ import annotations

import argparse
import uuid
from collections.abc import Callable
from typing import Any

from agentops_demo.agentcore.candidates import SUPPORTED_CANDIDATES, resolve_candidate
from agentops_demo.agentcore.request import CandidateSelector
from agentops_demo.contracts.agent_config import AgentConfig
from scripts.agentcore_package import load_config
from scripts.invoke_agentcore import invoke

Invocation = Callable[..., tuple[dict[str, Any], str]]
Input = Callable[[str], str]
Output = Callable[[str], None]
SessionIdFactory = Callable[[], str]


def _info(*, config: AgentConfig, candidate: CandidateSelector, session_id: str) -> str:
    return "\n".join(
        (
            f"Session: {session_id}",
            f"Candidate: {candidate}",
            f"Model: {config.model.provider}/{config.model.model_id}",
            f"Agent config fingerprint: {config.fingerprint()}",
            f"Source revision: {config.agent.source_revision}",
            "Last trace: not queried (trace lookup is not on the chat hot path)",
        )
    )


def _completed_tools(result: dict[str, Any]) -> list[str]:
    tools = result.get("completed_tools", [])
    if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        raise RuntimeError("AgentCore response has invalid completed_tools")
    return tools


def run_chat(
    *,
    config: AgentConfig | None = None,
    candidate: CandidateSelector = "bedrock",
    invoke_fn: Invocation = invoke,
    input_fn: Input = input,
    output_fn: Output = print,
    session_id_factory: SessionIdFactory = lambda: str(uuid.uuid4()),
) -> int:
    """Run terminal I/O for one AgentCore runtime session."""

    config = resolve_candidate(config or load_config(), candidate)
    session_id = session_id_factory()
    if not session_id.strip():
        raise ValueError("session ID factory returned a blank value")
    last_tools: list[str] = []
    output_fn("AgentCore interactive session")
    output_fn(f"Candidate: {candidate}")
    output_fn(f"Model: {config.model.provider}/{config.model.model_id}")
    output_fn(f"Session: {session_id}")
    output_fn("")
    while True:
        try:
            instruction = input_fn("you> ").strip()
        except EOFError:
            break
        if not instruction:
            continue
        command = instruction.lower()
        if command in {"/exit", "/quit"}:
            break
        if command == "/info":
            output_fn(_info(config=config, candidate=candidate, session_id=session_id))
            continue
        if command == "/tools":
            output_fn(f"tools> {', '.join(last_tools) if last_tools else 'none'}")
            continue
        if command.startswith("/"):
            output_fn(f"error> unknown command: {instruction}")
            continue
        try:
            result, returned_session_id = invoke_fn(
                instruction,
                session_id=session_id,
                candidate=candidate,
            )
            if returned_session_id != session_id:
                raise RuntimeError(
                    "AgentCore returned an unexpected runtime session ID: "
                    f"expected {session_id!r}, received {returned_session_id!r}"
                )
            final_response = result.get("final_response")
            if not isinstance(final_response, str):
                raise RuntimeError("AgentCore response has invalid final_response")
            last_tools = _completed_tools(result)
        except Exception as exc:
            output_fn(f"error> {exc}")
            continue
        output_fn(f"agent> {final_response}")
        if last_tools:
            output_fn(f"tools> {', '.join(last_tools)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=SUPPORTED_CANDIDATES, default="bedrock")
    args = parser.parse_args(argv)
    return run_chat(candidate=args.candidate)


if __name__ == "__main__":
    raise SystemExit(main())
