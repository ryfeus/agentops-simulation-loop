"""Offline guards for optional experiment transport and paid inference."""

from types import SimpleNamespace

import pytest

from experiments.decision_model import cloud
from scripts.aws_context import verified_session


@pytest.mark.parametrize("value", [None, "", "123", "abcdefghijkl"])
def test_transport_rejects_unconfigured_account_before_clients(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("EXPECTED_AWS_ACCOUNT_ID", raising=False)
    else:
        monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", value)
    calls = []
    monkeypatch.setattr("boto3.Session", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="EXPECTED_AWS_ACCOUNT_ID"):
        cloud.clients()
    assert calls == []


def test_transport_honors_profile_and_verifies_before_service_clients(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "research-sandbox")
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", "123456789012")
    calls = []

    def client(name, **kwargs):
        calls.append(name)
        return SimpleNamespace(get_caller_identity=lambda: {"Account": "123456789012"})

    def session(**kwargs):
        assert kwargs == {"profile_name": "research-sandbox", "region_name": "us-west-2"}
        return SimpleNamespace(client=client)

    monkeypatch.setattr("boto3.Session", session)
    cloud.clients()
    assert calls == ["sts", "ssm", "s3"]


def test_transport_mismatch_never_builds_mutating_clients(monkeypatch):
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", "123456789012")
    calls = []

    def client(name):
        calls.append(name)
        return SimpleNamespace(get_caller_identity=lambda: {"Account": "210987654321"})

    monkeypatch.setattr("boto3.Session", lambda **kwargs: SimpleNamespace(client=client))
    with pytest.raises(RuntimeError, match="refusing AWS"):
        cloud.clients()
    assert calls == ["sts"]


@pytest.mark.parametrize("account", ["123456789012", "210987654321"])
def test_bucket_name_uses_validated_operator_identity(account):
    # Assemble the fictional worker ID without introducing an operational receipt.
    worker = "i-" + "0123456789abcdef0"
    assert cloud.bucket_name(account, worker) == f"agentops-decider-{account}-{worker[2:]}"
    with pytest.raises(ValueError):
        cloud.bucket_name(account, "invalid-worker")
    with pytest.raises(ValueError):
        cloud.bucket_name("invalid-account", worker)


def test_fallback_rejects_account_before_model_construction(
    monkeypatch, tmp_path, agent_config_data
):
    import json

    from experiments.decision_model.harbor import llm_fallback

    monkeypatch.delenv("EXPECTED_AWS_ACCOUNT_ID", raising=False)
    config = tmp_path / "config.json"
    agent_config_data["model"] = {"provider": "bedrock", "model_id": "test-model"}
    config.write_text(json.dumps(agent_config_data))
    monkeypatch.setattr(
        "agentops_demo.agent.model.create_model", lambda *a, **k: pytest.fail("model constructed")
    )
    with pytest.raises(ValueError, match="EXPECTED_AWS_ACCOUNT_ID"):
        llm_fallback(config)


def test_account_guard_does_not_infer_expectation_from_caller(monkeypatch):
    monkeypatch.delenv("EXPECTED_AWS_ACCOUNT_ID", raising=False)
    with pytest.raises(ValueError):
        verified_session(session_factory=lambda **kwargs: pytest.fail("session created"))


def test_inference_uses_verified_session_not_ambient_credentials(monkeypatch, agent_config):
    import langchain_aws

    from agentops_demo.agent.model import create_model

    calls, received = [], []
    client = object()

    def service(name, **kwargs):
        calls.append((name, kwargs))
        return client

    monkeypatch.setenv("BILLING_MODEL_TOTAL_ATTEMPTS", "1")
    monkeypatch.setattr(
        langchain_aws, "ChatBedrockConverse", lambda **kwargs: received.append(kwargs)
    )
    create_model(agent_config, region_name="us-west-2", aws_session=SimpleNamespace(client=service))
    assert calls[0][0] == "bedrock-runtime"
    assert calls[0][1]["region_name"] == "us-west-2"
    assert calls[0][1]["config"].retries["total_max_attempts"] == 1
    assert [name for name, _ in calls] == ["bedrock-runtime", "bedrock"]
    assert received[0]["client"] is client
    assert received[0]["bedrock_client"] is client


@pytest.mark.asyncio
async def test_replay_rejects_account_before_paid_model_or_outputs(monkeypatch, tmp_path):
    import json

    from experiments.decision_model import llm

    monkeypatch.delenv("EXPECTED_AWS_ACCOUNT_ID", raising=False)
    source = tmp_path / "rows.jsonl"
    source.write_text(json.dumps({"split": "test"}) + "\n")
    output = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(
        "agentops_demo.agent.model.create_model", lambda *a, **k: pytest.fail("model constructed")
    )
    with pytest.raises(ValueError, match="EXPECTED_AWS_ACCOUNT_ID"):
        await llm.run(
            SimpleNamespace(data=source, split="test", limit=1, output=output, model="test-model")
        )
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
