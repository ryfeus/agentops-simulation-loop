from __future__ import annotations

from conftest import MCPServerFactory
from fastmcp import Client

from agentops_demo.billing.sqlite_repository import SQLiteBillingRepository


async def test_http_mcp_exposes_exact_tools_and_persists_calls(
    mcp_server_factory: MCPServerFactory,
) -> None:
    server = await mcp_server_factory("permissive")

    async with Client(server.url) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools} == {
            "get_invoice",
            "refund_invoice",
            "escalate_dispute",
        }

        invoice_result = await client.call_tool("get_invoice", {"invoice_id": "inv-123"})
        assert invoice_result.data["status"] == "disputed"

        refund_result = await client.call_tool(
            "refund_invoice",
            {"invoice_id": "inv-123", "reason": "duplicate charge"},
        )
        assert refund_result.data == {
            "success": True,
            "invoice_id": "inv-123",
            "reason": None,
        }

        escalation_result = await client.call_tool(
            "escalate_dispute",
            {"invoice_id": "inv-123", "reason": "specialist review"},
        )
        assert escalation_result.data["escalation_id"] == "escalation-inv-123"

    repository = SQLiteBillingRepository(server.database_path)
    assert await repository.mutation_counts("inv-123") == (1, 1)


async def test_http_mcp_returns_useful_domain_error(
    mcp_server_factory: MCPServerFactory,
) -> None:
    server = await mcp_server_factory("enforced")

    async with Client(server.url) as client:
        result = await client.call_tool(
            "refund_invoice",
            {"invoice_id": "inv-123", "reason": "duplicate charge"},
            raise_on_error=False,
        )

    assert result.is_error
    assert "disputed_invoice_requires_escalation" in str(result.content)
    repository = SQLiteBillingRepository(server.database_path)
    assert await repository.mutation_counts("inv-123") == (0, 0)


async def test_http_mcp_rejects_invalid_request(
    mcp_server_factory: MCPServerFactory,
) -> None:
    server = await mcp_server_factory()

    async with Client(server.url) as client:
        result = await client.call_tool(
            "refund_invoice",
            {"invoice_id": "inv-123", "reason": " "},
            raise_on_error=False,
        )

    assert result.is_error
    assert await SQLiteBillingRepository(server.database_path).mutation_counts("inv-123") == (
        0,
        0,
    )
