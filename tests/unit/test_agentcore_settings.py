from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentops_demo.agentcore.settings import AgentCoreSettings, load_agent_config


def test_load_agent_config_prefers_json(
    monkeypatch: pytest.MonkeyPatch, agent_config_data: dict[str, object]
) -> None:
    monkeypatch.setenv("AGENT_CONFIG_JSON", json.dumps(agent_config_data))
    monkeypatch.setenv("AGENT_CONFIG_PATH", "/missing")
    assert load_agent_config().agent.source_revision == "abc123"


def test_load_agent_config_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_config_data: dict[str, object],
) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(agent_config_data))
    monkeypatch.delenv("AGENT_CONFIG_JSON", raising=False)
    monkeypatch.setenv("AGENT_CONFIG_PATH", str(path))
    assert load_agent_config().model.provider == "bedrock"


def test_load_agent_config_requires_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_CONFIG_JSON", raising=False)
    monkeypatch.delenv("AGENT_CONFIG_PATH", raising=False)
    with pytest.raises(ValueError, match="must be set"):
        load_agent_config()


def test_agentcore_settings_url(agent_config) -> None:
    settings = AgentCoreSettings(config=agent_config, mcp_port=8123)
    assert settings.mcp_url == "http://127.0.0.1:8123/mcp"
