from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.models.agent.context import AgentContext

from agentops_demo.harbor.agent import LangGraphBillingAgent
from agentops_demo.harbor.provenance import sha256_file


def write_candidate(tmp_path: Path, agent_config_data: dict[str, object]) -> tuple[Path, Path]:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(agent_config_data))
    wheel = tmp_path / "candidate-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    return config, wheel


def create_agent(
    tmp_path: Path, agent_config_data: dict[str, object], model_name: str = "scripted/bad"
) -> LangGraphBillingAgent:
    agent_config_data["model"] = {"provider": "scripted", "model_id": "bad"}
    config, wheel = write_candidate(tmp_path, agent_config_data)
    return LangGraphBillingAgent(
        tmp_path / "logs",
        agent_config_path=str(config),
        package_path=str(wheel),
        model_name=model_name,
    )


def test_adapter_validates_inputs_and_model_identity(
    tmp_path: Path, agent_config_data: dict[str, object]
) -> None:
    agent = create_agent(tmp_path, agent_config_data)
    assert agent.agent_config.model.model_id == "bad"
    assert len(agent.agent_config_fingerprint) == 64
    assert agent.package_sha256 == sha256_file(agent.package_path)

    with pytest.raises(ValueError, match="does not match AgentConfig"):
        create_agent(tmp_path, agent_config_data, model_name="scripted/correct")


def test_adapter_rejects_missing_or_nonwheel_package(
    tmp_path: Path, agent_config_data: dict[str, object]
) -> None:
    agent_config_data["model"] = {"provider": "scripted", "model_id": "bad"}
    config, _ = write_candidate(tmp_path, agent_config_data)

    with pytest.raises(FileNotFoundError, match="candidate package"):
        LangGraphBillingAgent(
            tmp_path / "logs",
            agent_config_path=str(config),
            package_path=str(tmp_path / "missing.whl"),
            model_name="scripted/bad",
        )

    package = tmp_path / "candidate.zip"
    package.write_bytes(b"package")
    with pytest.raises(ValueError, match="wheel filename"):
        LangGraphBillingAgent(
            tmp_path / "logs",
            agent_config_path=str(config),
            package_path=str(package),
            model_name="scripted/bad",
        )


async def test_adapter_install_uploads_local_inputs_and_starts_mcp(
    tmp_path: Path, agent_config_data: dict[str, object]
) -> None:
    agent = create_agent(tmp_path, agent_config_data)
    environment = SimpleNamespace(upload_file=AsyncMock())
    agent.exec_as_agent = AsyncMock()

    await agent.install(environment)

    assert environment.upload_file.await_count == 2
    assert (
        environment.upload_file.await_args_list[0]
        .args[1]
        .endswith("/candidate-0.1.0-py3-none-any.whl")
    )
    commands = [call.kwargs["command"] for call in agent.exec_as_agent.await_args_list]
    assert any("pip install --no-deps" in command for command in commands)
    assert any(
        "provenance.json" in command and agent.package_sha256 in command for command in commands
    )
    assert any("billing_server" in command and "permissive" in command for command in commands)
    assert any("/health" in command and "status==200" in command for command in commands)


async def test_adapter_run_passes_only_candidate_inputs(
    tmp_path: Path, agent_config_data: dict[str, object]
) -> None:
    agent = create_agent(tmp_path, agent_config_data)
    environment = SimpleNamespace(upload_file=AsyncMock())
    agent.exec_as_agent = AsyncMock()
    context = AgentContext()

    await agent.run("instruction", environment, context)

    command = agent.exec_as_agent.await_args.kwargs["command"]
    assert "agentops_demo.cli.run_once" in command
    assert "/logs/agent/result.json" in command
    assert "verifier" not in command
    assert "reward" not in command
    assert "invariant" not in command
    assert context.metadata is not None
    assert context.metadata == {
        "agent_config_fingerprint": agent.agent_config.fingerprint(),
        "source_revision": agent.agent_config.agent.source_revision,
        "package_sha256": sha256_file(agent.package_path),
        "model": "scripted/bad",
    }
