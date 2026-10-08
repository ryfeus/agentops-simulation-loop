"""FastMCP HTTP server exposing the Phase 0 billing tool contract."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from agentops_demo.billing.repository_factory import create_billing_repository
from agentops_demo.billing.service import BillingService


@dataclass(frozen=True)
class BillingServerSettings:
    """Environment-backed local MCP settings."""

    database_path: Path = Path("data/billing.db")
    repository_backend: str = "sqlite"
    policy_mode: str = "permissive"
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_environment(cls) -> BillingServerSettings:
        try:
            port = int(os.getenv("BILLING_MCP_PORT", "8000"))
        except ValueError as exc:
            raise ValueError("BILLING_MCP_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("BILLING_MCP_PORT must be between 1 and 65535")
        return cls(
            database_path=Path(os.getenv("BILLING_DATABASE_PATH", "data/billing.db")),
            repository_backend=os.getenv("BILLING_REPOSITORY_BACKEND", "sqlite"),
            policy_mode=os.getenv("BILLING_POLICY_MODE", "permissive"),
            host=os.getenv("BILLING_MCP_HOST", "127.0.0.1"),
            port=port,
        )


def create_billing_server(service: BillingService) -> FastMCP:
    """Create a side-effect-free billing server around a supplied service."""

    server = FastMCP("billing", mask_error_details=False, strict_input_validation=True)

    @server.tool
    async def get_invoice(invoice_id: str) -> dict[str, Any] | None:
        """Get an invoice by its domain identifier."""

        invoice = await service.get_invoice(invoice_id)
        return invoice.model_dump(mode="json") if invoice is not None else None

    @server.tool
    async def refund_invoice(invoice_id: str, reason: str) -> dict[str, Any]:
        """Create a refund for an invoice when the configured policy permits it."""

        result = await service.refund_invoice(invoice_id, reason)
        return result.model_dump(mode="json")

    @server.tool
    async def escalate_dispute(invoice_id: str, reason: str) -> dict[str, Any]:
        """Escalate an invoice dispute for specialist review."""

        result = await service.escalate_dispute(invoice_id, reason)
        return result.model_dump(mode="json")

    @server.custom_route("/health", methods=["GET"], include_in_schema=False)
    async def health(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("OK")

    return server


def create_server_from_settings(settings: BillingServerSettings) -> FastMCP:
    repository = create_billing_repository(
        backend=settings.repository_backend,
        sqlite_path=settings.database_path,
    )
    service = BillingService(repository, policy_mode=settings.policy_mode)
    return create_billing_server(service)


settings = BillingServerSettings.from_environment()
mcp = create_server_from_settings(settings)


def main() -> None:
    mcp.run(
        transport="http",
        host=settings.host,
        port=settings.port,
        show_banner=False,
    )


if __name__ == "__main__":
    main()
