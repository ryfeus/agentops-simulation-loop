"""Environment-backed AgentCore runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from agentops_demo.agent.graph import validate_runtime_config
from agentops_demo.contracts.agent_config import AgentConfig


def _boolean_environment(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def load_agent_config() -> AgentConfig:
    """Load one authoritative, complete configuration from JSON or a local path."""

    serialized = os.getenv("AGENT_CONFIG_JSON")
    if serialized is not None:
        if not serialized.strip():
            raise ValueError("AGENT_CONFIG_JSON must not be blank")
        config = AgentConfig.model_validate_json(serialized)
    else:
        path_value = os.getenv("AGENT_CONFIG_PATH", "").strip()
        if not path_value:
            raise ValueError("AGENT_CONFIG_JSON or AGENT_CONFIG_PATH must be set")
        path = Path(path_value)
        try:
            config = AgentConfig.model_validate_json(path.read_text())
        except OSError as exc:
            raise ValueError(f"cannot read AGENT_CONFIG_PATH {path}: {exc}") from exc
    validate_runtime_config(config)
    return config


@dataclass(frozen=True)
class AgentCoreSettings:
    config: AgentConfig
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8000
    mcp_startup_timeout: float = 30.0
    trace_content_enabled: bool = False
    world_snapshot_enabled: bool = False
    world_snapshot_required: bool = False
    world_snapshot_max_bytes: int = 32_768

    def __post_init__(self) -> None:
        validate_runtime_config(self.config)
        if not self.mcp_host.strip():
            raise ValueError("Billing MCP host must be nonblank")
        if not 1 <= self.mcp_port <= 65535:
            raise ValueError("Billing MCP port must be between 1 and 65535")
        if self.mcp_startup_timeout <= 0:
            raise ValueError("Billing MCP startup timeout must be positive")
        if self.world_snapshot_required and not self.world_snapshot_enabled:
            raise ValueError("world snapshot cannot be required when disabled")
        if self.world_snapshot_max_bytes <= 0:
            raise ValueError("world snapshot maximum size must be positive")

    @property
    def mcp_url(self) -> str:
        return f"http://{self.mcp_host}:{self.mcp_port}/mcp"

    @classmethod
    def from_environment(cls) -> AgentCoreSettings:
        try:
            port = int(os.getenv("BILLING_MCP_PORT", "8000"))
            timeout = float(os.getenv("BILLING_MCP_STARTUP_TIMEOUT", "30"))
            snapshot_max_bytes = int(os.getenv("AGENTOPS_WORLD_SNAPSHOT_MAX_BYTES", "32768"))
        except ValueError as exc:
            raise ValueError("MCP port and startup timeout must be numeric") from exc
        return cls(
            config=load_agent_config(),
            mcp_host=os.getenv("BILLING_MCP_HOST", "127.0.0.1"),
            mcp_port=port,
            mcp_startup_timeout=timeout,
            trace_content_enabled=_boolean_environment("AGENTOPS_TRACE_CONTENT_ENABLED", False),
            world_snapshot_enabled=_boolean_environment("AGENTOPS_WORLD_SNAPSHOT_ENABLED", False),
            world_snapshot_required=_boolean_environment("AGENTOPS_WORLD_SNAPSHOT_REQUIRED", False),
            world_snapshot_max_bytes=snapshot_max_bytes,
        )
