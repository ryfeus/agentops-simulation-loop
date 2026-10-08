from __future__ import annotations

import pytest
from conftest import MCPServerFactory
from langchain_core.messages import ToolMessage

from agentops_demo.agent.graph import build_agent
from agentops_demo.agent.scripted import create_scripted_model
from agentops_demo.billing.sqlite_repository import SQLiteBillingRepository
from agentops_demo.contracts.agent_config import AgentConfig


@pytest.mark.parametrize(
    ("mode", "prompt_version", "expected_tools", "expected_counts", "expected_status"),
    [
        (
            "bad",
            "billing-v1",
            ["get_invoice", "refund_invoice"],
            (1, 0),
            "refunded",
        ),
        (
            "correct",
            "billing-v2",
            ["get_invoice", "escalate_dispute"],
            (0, 1),
            "disputed",
        ),
    ],
)
async def test_scripted_langgraph_agent_crosses_real_mcp_http_boundary(
    mcp_server_factory: MCPServerFactory,
    mode: str,
    prompt_version: str,
    expected_tools: list[str],
    expected_counts: tuple[int, int],
    expected_status: str,
) -> None:
    server = await mcp_server_factory("permissive")
    config = AgentConfig.model_validate(
        {
            "agent": {"source_revision": "integration-test"},
            "model": {"provider": "scripted", "model_id": mode},
            "prompt": {"version": prompt_version},
            "tools": {"version": "billing-mcp-v1"},
            "harness": {"framework": "langgraph", "version": "v1"},
        }
    )
    graph = await build_agent(
        config=config,
        model=create_scripted_model(mode),  # type: ignore[arg-type]
        mcp_url=server.url,
    )

    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "I was charged twice for April. Please refund invoice inv-123.",
                }
            ]
        }
    )

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [message.name for message in tool_messages] == expected_tools
    assert result["messages"][-1].content

    repository = SQLiteBillingRepository(server.database_path)
    assert await repository.mutation_counts("inv-123") == expected_counts
    invoice = await repository.get_invoice("inv-123")
    assert invoice is not None
    assert invoice.status == expected_status
