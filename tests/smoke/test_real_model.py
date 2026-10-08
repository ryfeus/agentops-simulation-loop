from __future__ import annotations

import os

import pytest
from conftest import MCPServerFactory
from langchain_core.messages import AIMessage, ToolMessage

from agentops_demo.agent.graph import build_agent
from agentops_demo.agent.model import create_model
from agentops_demo.contracts.agent_config import AgentConfig

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        os.getenv("RUN_REAL_MODEL_TESTS") != "1",
        reason="set RUN_REAL_MODEL_TESTS=1 to enable the Bedrock smoke test",
    ),
]


async def test_bedrock_agent_can_reach_local_mcp(
    mcp_server_factory: MCPServerFactory,
) -> None:
    model_id = os.getenv("BEDROCK_MODEL_ID")
    if not model_id:
        pytest.fail("BEDROCK_MODEL_ID is required when RUN_REAL_MODEL_TESTS=1")
    server = await mcp_server_factory("permissive")
    config = AgentConfig.model_validate(
        {
            "agent": {"source_revision": "local-smoke"},
            "model": {"provider": "bedrock", "model_id": model_id},
            "prompt": {"version": "billing-v1"},
            "tools": {"version": "billing-mcp-v1"},
            "harness": {"framework": "langgraph", "version": "v1"},
        }
    )
    graph = await build_agent(
        config=config,
        model=create_model(config, region_name=os.getenv("AWS_REGION")),
        mcp_url=server.url,
    )

    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Inspect invoice inv-123 and explain its current status.",
                }
            ]
        }
    )

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert any(message.name == "get_invoice" for message in tool_messages)
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content
