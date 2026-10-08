from __future__ import annotations

from pathlib import Path

import pytest

from agentops_demo.mcp.billing_server import BillingServerSettings


def test_server_settings_have_safe_local_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "BILLING_DATABASE_PATH",
        "BILLING_POLICY_MODE",
        "BILLING_MCP_HOST",
        "BILLING_MCP_PORT",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = BillingServerSettings.from_environment()

    assert settings.database_path == Path("data/billing.db")
    assert settings.policy_mode == "permissive"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000


def test_server_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BILLING_DATABASE_PATH", "/tmp/test-billing.db")
    monkeypatch.setenv("BILLING_POLICY_MODE", "enforced")
    monkeypatch.setenv("BILLING_MCP_HOST", "localhost")
    monkeypatch.setenv("BILLING_MCP_PORT", "8123")

    settings = BillingServerSettings.from_environment()

    assert settings.database_path == Path("/tmp/test-billing.db")
    assert settings.policy_mode == "enforced"
    assert settings.host == "localhost"
    assert settings.port == 8123


@pytest.mark.parametrize("port", ["not-a-port", "0", "65536"])
def test_invalid_server_port_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    port: str,
) -> None:
    monkeypatch.setenv("BILLING_MCP_PORT", port)

    with pytest.raises(ValueError, match="BILLING_MCP_PORT"):
        BillingServerSettings.from_environment()
