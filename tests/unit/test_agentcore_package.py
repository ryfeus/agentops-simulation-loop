from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import agentcore_package


def test_production_and_ci_share_arm64_docker_command() -> None:
    wheel = Path(".agentcore/package/project.whl")
    production = agentcore_package.docker_build_command(
        wheel=wheel, tag="registry/repository:revision", push=True
    )
    build_only = agentcore_package.docker_build_command(
        wheel=wheel, tag="agentops-demo-agentcore:image-check", push=False
    )
    for command in (production, build_only):
        assert command[:3] == ["docker", "buildx", "build"]
        assert command[command.index("--platform") + 1] == "linux/arm64"
        assert command[command.index("--file") + 1] == "deploy/agentcore/Dockerfile"
        assert f"WHEEL_FILE={wheel}" in command
        assert "RUNTIME_REQUIREMENTS=deploy/agentcore/requirements.txt" in command
    assert "--push" in production
    assert not any(argument.startswith("--output") for argument in production)
    assert "--output=type=cacheonly" in build_only
    assert "--push" not in build_only
    dockerfile = Path("deploy/agentcore/Dockerfile").read_text()
    assert 'CMD ["opentelemetry-instrument", "uvicorn"' in dockerfile
    assert "OPENINFERENCE_HIDE_INPUTS=true" in dockerfile


def test_runtime_requirements_have_exactly_one_langchain_instrumentor() -> None:
    requirements = Path("deploy/agentcore/requirements.txt").read_text().splitlines()
    names = {line.split("==", 1)[0] for line in requirements if "==" in line}
    assert "aws-opentelemetry-distro" in names
    assert {name for name in names if "instrumentation-langchain" in name} == {
        "openinference-instrumentation-langchain"
    }


def test_build_only_uses_no_config_terraform_or_aws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "project.whl"
    calls: list[list[str]] = []
    monkeypatch.setattr(agentcore_package, "IMAGE_CHECK_DIR", tmp_path)
    monkeypatch.setattr(agentcore_package, "verify_runtime_requirements", lambda: None)
    monkeypatch.setattr(agentcore_package, "build_wheel", lambda _output: wheel)
    monkeypatch.setattr(
        agentcore_package,
        "load_config",
        lambda: (_ for _ in ()).throw(AssertionError("config must not be read")),
    )
    monkeypatch.setattr(
        agentcore_package,
        "terraform_outputs",
        lambda: (_ for _ in ()).throw(AssertionError("Terraform must not be queried")),
    )

    def run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(agentcore_package.subprocess, "run", run)
    assert agentcore_package.build_image_check() == wheel
    assert len(calls) == 1
    assert calls[0][-2:] == ["--output=type=cacheonly", "."]
    assert not any("aws" in argument or "ecr" in argument for argument in calls[0])


def test_registry_publish_rejects_missing_expected_account(monkeypatch):
    monkeypatch.delenv("EXPECTED_AWS_ACCOUNT_ID", raising=False)
    with pytest.raises(ValueError, match="EXPECTED_AWS_ACCOUNT_ID"):
        agentcore_package.build_and_push()


def test_registry_subprocesses_use_verified_profile(monkeypatch, tmp_path, agent_config):
    from scripts import aws_context

    monkeypatch.setenv("AWS_PROFILE", "image-sandbox")
    monkeypatch.setattr(aws_context, "verified_session", lambda: None)
    monkeypatch.setattr(agentcore_package, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(
        agentcore_package, "current_revision", lambda: agent_config.agent.source_revision
    )
    monkeypatch.setattr(agentcore_package, "load_config", lambda: agent_config)
    monkeypatch.setattr(agentcore_package, "verify_runtime_requirements", lambda: None)
    wheel = tmp_path / "synthetic.whl"
    wheel.write_bytes(b"synthetic wheel")
    monkeypatch.setattr(agentcore_package, "build_wheel", lambda _: wheel)
    monkeypatch.setattr(agentcore_package, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(
        agentcore_package,
        "terraform_outputs",
        lambda: {
            "ecr_repository_url": "123456789012.dkr.ecr.us-west-2.amazonaws.com/synthetic",
            "ecr_repository_name": "synthetic",
        },
    )
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        output = "sha256:" + "a" * 64 if "describe-images" in command else b"synthetic password"
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(agentcore_package.subprocess, "run", execute)
    agentcore_package.build_and_push()
    aws_commands = [command for command in commands if command[0] == "aws"]
    assert len(aws_commands) == 2
    assert all(command[1:3] == ("--profile", "image-sandbox") for command in aws_commands)
