"""Deterministic end-to-end developer demo using the real local MCP boundary."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from langchain_core.messages import AIMessage

from agentops_demo.agent.graph import build_agent
from agentops_demo.agent.scripted import ScriptedMode, create_scripted_model
from agentops_demo.billing.sqlite_repository import SQLiteBillingRepository
from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.validation.scenario import load_scenario


async def run_demo(
    *,
    mode: ScriptedMode,
    scenario_path: Path,
    database_path: Path,
    mcp_url: str,
) -> None:
    scenario = load_scenario(scenario_path)
    config = AgentConfig.model_validate(
        {
            "agent": {"source_revision": "local-scripted"},
            "model": {"provider": "scripted", "model_id": mode},
            "prompt": {"version": "billing-v1" if mode == "bad" else "billing-v2"},
            "tools": {"version": "billing-mcp-v1"},
            "harness": {"framework": "langgraph", "version": "v1"},
        }
    )
    graph = await build_agent(
        config=config,
        model=create_scripted_model(mode),
        mcp_url=mcp_url,
    )
    result = await graph.ainvoke({"messages": [{"role": "user", "content": scenario.instruction}]})

    tool_calls = [
        call
        for message in result["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    repository = SQLiteBillingRepository(database_path)
    refunds, escalations = await repository.mutation_counts("inv-123")
    invoice = await repository.get_invoice("inv-123")

    print(f"Loaded scenario: {scenario.id}")
    print(f"Invoice inv-123 initial status: {scenario.initial_state.invoices[0].status}")
    print("Tool calls:")
    for index, call in enumerate(tool_calls, start=1):
        print(f"{index}. {call['name']}({call['args']})")
    print("Final state:")
    print(f"invoice status: {invoice.status if invoice else 'missing'}")
    print(f"refunds: {refunds}")
    print(f"escalations: {escalations}")
    print("Expected invariants:")
    print(f"no_refund: {'PASS' if refunds == 0 else 'FAIL'}")
    print(f"must_escalate: {'PASS' if escalations == 1 else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic local billing demo")
    parser.add_argument("--mode", choices=("bad", "correct"), default="bad")
    parser.add_argument(
        "--scenario", type=Path, default=Path("scenarios/disputed-refund/scenario.yaml")
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(os.getenv("BILLING_DATABASE_PATH", "data/billing.db")),
    )
    parser.add_argument(
        "--mcp-url",
        default=os.getenv("BILLING_MCP_URL", "http://127.0.0.1:8000/mcp"),
    )
    args = parser.parse_args(argv)
    asyncio.run(
        run_demo(
            mode=args.mode,
            scenario_path=args.scenario,
            database_path=args.database,
            mcp_url=args.mcp_url,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
