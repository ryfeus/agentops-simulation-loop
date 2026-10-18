from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.harbor.trajectory import write_trajectory
from scripts import local_harbor_demo as demo


@pytest.mark.parametrize("before_prompt", ["billing-v1", "billing-refund-v1"])
def test_prepare_writes_private_token_and_cleanup_preserves_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, before_prompt: str
) -> None:
    source = Path("benchmarks/disputed-refund").resolve()
    monkeypatch.chdir(tmp_path)
    client = Mock()
    client.get_caller_identity.return_value = {"Account": "123456789012"}
    monkeypatch.setattr(demo, "verified_session", lambda: SimpleNamespace(client=lambda _: client))
    provider = Mock()
    provider.load.return_value = object()
    monkeypatch.setattr("botocore.credentials.EnvProvider", lambda: provider)
    generate = Mock(return_value="secret-test-token")
    monkeypatch.setitem(
        sys.modules, "aws_bedrock_token_generator", SimpleNamespace(provide_token=generate)
    )
    monkeypatch.setattr(
        demo,
        "build_package",
        lambda *_args, **_kwargs: (
            tmp_path / "candidate.whl",
            {"sha256": "a" * 64, "source_revision": "development-test"},
        ),
    )

    output = demo.prepare(source, development=True, before_prompt=before_prompt)

    manifest = json.loads((output / "comparison.json").read_text())
    token_path = Path(manifest["token_path"])
    assert token_path.read_text() == "secret-test-token"
    assert token_path.stat().st_mode & 0o777 == 0o600
    assert token_path.parent.stat().st_mode & 0o777 == 0o700
    assert manifest["execution_profile"]["credential_mode"] == "bedrock-bearer-token"
    assert "secret-test-token" not in (output / "comparison.json").read_text()
    command = demo.harbor_command(output, manifest, "before")
    assert "bedrock_token_path=" + str(token_path) in command
    assert "proxy" not in repr(command)
    assert "secret-test-token" not in repr(command)
    before = json.loads((output / "before-config.json").read_text())
    after = json.loads((output / "after-config.json").read_text())
    assert before.pop("prompt") == {"version": before_prompt}
    assert after.pop("prompt") == {"version": "billing-v2"}
    assert before == after
    demo.cleanup(output)
    demo.cleanup(output)
    assert not token_path.exists()
    assert (output / "before-config.json").exists()
    assert json.loads((output / "comparison.json").read_text())["host_token_removed"] is True


def test_missing_token_refuses_model_run(tmp_path: Path) -> None:
    demo.save(tmp_path / "comparison.json", {"token_path": str(tmp_path / "missing")})
    with pytest.raises(RuntimeError, match="missing or expired"):
        demo.run_trial(tmp_path, "before")


def test_verify_accepts_retained_job_without_reinvoking_model(
    tmp_path: Path, agent_config: AgentConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "jobs/before"
    trial = job / "trial"
    (trial / "agent").mkdir(parents=True)
    demo.save(job / "result.json", {"n_total_trials": 1})
    demo.save(
        trial / "result.json",
        {
            "trial_uri": trial.as_uri(),
            "verifier_result": {"rewards": {"reward": 0.0}},
            "exception_info": None,
        },
    )
    demo.save(
        trial / "agent/result.json",
        {"tool_calls": [], "agent_config_fingerprint": agent_config.fingerprint()},
    )
    demo.save(
        trial / "agent/provenance.json",
        {
            "agent_config_fingerprint": agent_config.fingerprint(),
            "package_sha256": "a" * 64,
            "source_revision": agent_config.agent.source_revision,
            "model": "bedrock/model-v1",
        },
    )
    (tmp_path / "before-config.json").write_text(agent_config.model_dump_json())
    write_trajectory(
        trial / "agent/trajectory.json",
        config=agent_config,
        instruction="test",
        messages=[
            AIMessage(
                content="test", response_metadata={"ResponseMetadata": {"RequestId": "request-id"}}
            )
        ],
    )
    monkeypatch.setattr(demo, "harbor_task_sha256", lambda _: "b" * 64)
    manifest = {
        "results": {"before": {}},
        "model": "bedrock/model-v1",
        "package": {"sha256": "a" * 64},
        "source_task": "unused",
        "base_task_sha256": "b" * 64,
    }

    demo.verify_trial(tmp_path, manifest, "before")

    assert manifest["results"]["before"]["accepted"] is True
    assert manifest["results"]["before"]["observed"]["reward"] == 0
    assert manifest["results"]["before"]["bedrock_calls"][0]["request_id"] == "request-id"


def test_harbor_subprocess_gets_token_path_without_host_aws_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text("private-bearer-token")
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"test wheel")
    demo.save(
        tmp_path / "comparison.json",
        {
            "token_path": str(token),
            "token_not_after_utc": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            "wheel": str(wheel),
            "package": {"sha256": demo.sha256_file(wheel)},
            "model": "bedrock/model-v1",
            "results": {},
        },
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "host-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "host-secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "host-session-token")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "unrelated-host-bearer-token")
    run = Mock(return_value=SimpleNamespace(returncode=9))
    monkeypatch.setattr(demo.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Harbor process failed"):
        demo.run_trial(tmp_path, "before")
    assert not any(name.startswith("AWS_") for name in run.call_args.kwargs["env"])
    assert "private-bearer-token" not in repr(run.call_args)
    manifest = json.loads((tmp_path / "comparison.json").read_text())
    assert manifest["results"]["before"]["returncode"] == 9
    assert manifest["results"]["before"]["accepted"] is False
