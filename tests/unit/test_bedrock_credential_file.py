from __future__ import annotations

import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import aws_context
from scripts import benchmark_billing_bedrock as benchmark


@pytest.mark.parametrize("fail", [False, True])
def test_ephemeral_credentials_are_private_and_removed(monkeypatch, tmp_path, agent_config, fail):
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", "123456789012")
    monkeypatch.setattr(
        aws_context, "verified_session", lambda: SimpleNamespace(client=lambda _: None)
    )
    config_path = tmp_path / "candidate.json"
    config_path.write_text(agent_config.model_dump_json())
    monkeypatch.setattr(
        benchmark,
        "render_catalog_sync",
        lambda *args: [SimpleNamespace(scenario=SimpleNamespace(id="synthetic"))],
    )
    monkeypatch.setattr(benchmark, "derive_bedrock_task", lambda *args: None)
    monkeypatch.setattr(
        benchmark,
        "assume_bedrock_role",
        lambda *args, **kwargs: SimpleNamespace(
            credentials_file=lambda: "synthetic temporary credential fixture"
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "build_clean_wheel",
        lambda *args: (config_path, SimpleNamespace(sha256="a" * 64)),
    )
    credential_paths = []

    def execute(command, **kwargs):
        path = Path(
            next(
                item.split("=", 1)[1]
                for item in command
                if item.startswith("bedrock_credentials_path=")
            )
        )
        credential_paths.append(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        if fail:
            raise RuntimeError("trial failed")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", execute)
    arguments = {
        "catalog": tmp_path,
        "candidate_config": config_path,
        "role_arn": "arn:aws:iam::123456789012:role/synthetic",
        "output": tmp_path / "output",
    }
    if fail:
        with pytest.raises(RuntimeError, match="trial failed"):
            benchmark.run(**arguments)
    else:
        benchmark.run(**arguments)
    assert len(credential_paths) == 1
    assert not credential_paths[0].exists()


@pytest.mark.asyncio
async def test_dsql_connect_rejects_missing_account_before_connector(monkeypatch):
    from scripts.dsql_admin import connect_admin

    monkeypatch.delenv("EXPECTED_AWS_ACCOUNT_ID", raising=False)
    with pytest.raises(ValueError, match="EXPECTED_AWS_ACCOUNT_ID"):
        await connect_admin("synthetic.invalid", "us-west-2")


def test_agentcore_invocation_rejects_missing_account(monkeypatch):
    from scripts.invoke_agentcore import invoke

    monkeypatch.delenv("EXPECTED_AWS_ACCOUNT_ID", raising=False)
    with pytest.raises(ValueError, match="EXPECTED_AWS_ACCOUNT_ID"):
        invoke("synthetic instruction")
