from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zipfile import ZipFile

import pytest
from harbor.models.agent.context import AgentContext

from agentops_demo.harbor.agent import LangGraphBillingAgent
from agentops_demo.harbor.provenance import sha256_file


def write_candidate(tmp_path: Path, agent_config_data: dict[str, object]) -> tuple[Path, Path]:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(agent_config_data))
    wheel = tmp_path / "candidate-0.1.0-py3-none-any.whl"
    with ZipFile(wheel, "w") as package:
        package.writestr("agentops_demo/cli/run_once.py", 'parser.add_argument("--trajectory")')
        package.writestr("agentops_demo/harbor/trajectory.py", "")
        package.writestr(
            "agentops_demo/agent/model.py", 'TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK_FILE"'
        )
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


async def test_explicit_enforced_policy_and_inference_reach_sandbox(
    tmp_path: Path, agent_config_data: dict[str, object]
) -> None:
    agent_config_data["model"] = {"provider": "scripted", "model_id": "bad"}
    config, wheel = write_candidate(tmp_path, agent_config_data)
    agent = LangGraphBillingAgent(
        tmp_path / "logs",
        agent_config_path=str(config),
        package_path=str(wheel),
        model_name="scripted/bad",
        billing_policy_mode="enforced",
        inference_temperature=0,
        inference_max_tokens=1024,
        inference_total_attempts=1,
    )
    environment = SimpleNamespace(upload_file=AsyncMock())
    agent.exec_as_agent = AsyncMock()
    await agent.install(environment)
    assert any(
        "BILLING_POLICY_MODE=enforced" in call.kwargs["command"]
        for call in agent.exec_as_agent.await_args_list
    )
    await agent.run("instruction", environment, AgentContext())
    assert agent.exec_as_agent.await_args.kwargs["env"] == {
        "BILLING_MODEL_TEMPERATURE": "0.0",
        "BILLING_MODEL_MAX_TOKENS": "1024",
        "BILLING_MODEL_TOTAL_ATTEMPTS": "1",
    }


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
    assert "--trajectory /logs/agent/trajectory.json" in command
    assert agent.SUPPORTS_ATIF is True
    assert "verifier" not in command
    assert "reward" not in command
    assert "invariant" not in command
    assert context.metadata is not None
    assert context.metadata == {
        "agent_config_fingerprint": agent.agent_config.fingerprint(),
        "source_revision": agent.agent_config.agent.source_revision,
        "package_sha256": sha256_file(agent.package_path),
        "model": "scripted/bad",
        "billing_policy_mode": "permissive",
        "inference_environment": {},
    }


@pytest.mark.parametrize("has_exporter", [False, True])
def test_adapter_rejects_stale_wheel_before_container_setup(
    tmp_path: Path, agent_config_data: dict[str, object], has_exporter: bool
) -> None:
    agent_config_data["model"] = {"provider": "scripted", "model_id": "bad"}
    config, wheel = write_candidate(tmp_path, agent_config_data)
    with ZipFile(wheel, "w") as package:
        package.writestr("agentops_demo/cli/run_once.py", 'parser.add_argument("--output")')
        if has_exporter:
            package.writestr("agentops_demo/harbor/trajectory.py", "")
    with pytest.raises(ValueError, match="rebuild the wheel and restart demo setup"):
        LangGraphBillingAgent(
            tmp_path / "logs",
            agent_config_path=str(config),
            package_path=str(wheel),
            model_name="scripted/bad",
        )


async def test_adapter_creates_private_credential_directory_before_upload(
    tmp_path: Path, agent_config_data: dict[str, object]
) -> None:
    agent = create_agent(tmp_path, agent_config_data)
    credentials = tmp_path / "credentials"
    credentials.write_text("test placeholder")
    agent.bedrock_credentials_path = credentials
    events: list[str] = []

    async def upload(source: Path, target: str) -> None:
        events.append("upload:" + target)

    async def execute(environment: object, *, command: str, **kwargs: object) -> None:
        events.append(command)

    environment = SimpleNamespace(upload_file=AsyncMock(side_effect=upload))
    agent.exec_as_agent = AsyncMock(side_effect=execute)
    await agent.install(environment)
    assert events.index(
        "mkdir -p /installed-agent/aws && chmod 700 /installed-agent/aws"
    ) < events.index("upload:/installed-agent/aws/credentials")
    assert "chmod 600 /installed-agent/aws/credentials" in events


async def test_bearer_token_is_private_and_removed_even_when_invocation_fails(
    tmp_path: Path, agent_config_data: dict[str, object]
) -> None:
    agent_config_data["model"] = {
        "provider": "bedrock",
        "model_id": "us.anthropic.claude-sonnet-4-6",
    }
    config, wheel = write_candidate(tmp_path, agent_config_data)
    token_path = tmp_path / "token"
    token_path.write_text("secret-placeholder")
    agent = LangGraphBillingAgent(
        tmp_path / "logs",
        agent_config_path=str(config),
        package_path=str(wheel),
        bedrock_token_path=str(token_path),
        model_name="bedrock/us.anthropic.claude-sonnet-4-6",
    )
    environment = SimpleNamespace(upload_file=AsyncMock())
    agent.exec_as_agent = AsyncMock()
    await agent.install(environment)
    assert environment.upload_file.await_args.args == (
        token_path,
        "/installed-agent/aws/bedrock-token",
    )
    assert (
        agent.exec_as_agent.await_args.kwargs["command"]
        == "chmod 600 /installed-agent/aws/bedrock-token"
    )

    agent.exec_as_agent.reset_mock()
    agent.exec_as_agent.side_effect = [RuntimeError("invocation failed"), None]
    with pytest.raises(RuntimeError, match="invocation failed"):
        await agent.run("instruction", environment, AgentContext())
    invocation, cleanup = agent.exec_as_agent.await_args_list
    assert (
        invocation.kwargs["env"]["AWS_BEARER_TOKEN_BEDROCK_FILE"]
        == "/installed-agent/aws/bedrock-token"
    )
    assert "AWS_SHARED_CREDENTIALS_FILE" not in invocation.kwargs["env"]
    assert "secret-placeholder" not in repr(invocation)
    assert cleanup.kwargs["command"] == "rm -f /installed-agent/aws/bedrock-token"


def test_adapter_rejects_ambiguous_credentials(
    tmp_path: Path, agent_config_data: dict[str, object]
) -> None:
    config, wheel = write_candidate(tmp_path, agent_config_data)
    with pytest.raises(ValueError, match="choose either"):
        LangGraphBillingAgent(
            tmp_path / "logs",
            agent_config_path=str(config),
            package_path=str(wheel),
            bedrock_credentials_path="credentials",
            bedrock_token_path="token",
        )
