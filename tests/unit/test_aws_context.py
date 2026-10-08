from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.aws_context import expected_account_id, require_account, verified_session


@pytest.mark.parametrize("value", ["", "123", "abcdefghijkl", "12345678901x", chr(0xFF11) * 12])
def test_invalid_expected_account_never_loads_session(monkeypatch, value):
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", value)
    calls = []
    with pytest.raises(ValueError, match="exactly 12 digits"):
        verified_session(session_factory=lambda **kwargs: calls.append(kwargs))
    assert not calls


def test_missing_expected_account_never_calls_sts(monkeypatch):
    monkeypatch.delenv("EXPECTED_AWS_ACCOUNT_ID", raising=False)
    with pytest.raises(ValueError):
        require_account(SimpleNamespace(get_caller_identity=lambda: pytest.fail("STS called")))
    assert expected_account_id({"EXPECTED_AWS_ACCOUNT_ID": "123456789012"}) == "123456789012"
    with pytest.raises(ValueError):
        expected_account_id({})


@pytest.mark.parametrize("account", ["123456789012", "210987654321"])
def test_profile_and_account_are_operator_selected(monkeypatch, account):
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", account)
    monkeypatch.setenv("AWS_PROFILE", "research-sandbox")
    calls = []
    session = SimpleNamespace(
        client=lambda service: SimpleNamespace(get_caller_identity=lambda: {"Account": account})
    )

    def factory(**kwargs):
        calls.append(kwargs)
        return session

    assert verified_session(session_factory=factory) is session
    assert calls == [{"profile_name": "research-sandbox", "region_name": "us-west-2"}]


def test_mismatch_never_creates_resource_client(monkeypatch):
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", "123456789012")
    calls = []

    def client(service):
        calls.append(service)
        return SimpleNamespace(get_caller_identity=lambda: {"Account": "210987654321"})

    with pytest.raises(RuntimeError, match="refusing AWS"):
        verified_session(session=SimpleNamespace(client=client))
    assert calls == ["sts"]


def test_phase11_recipes_use_shell_guard():
    makefile = Path("Makefile").read_text()
    for name in ("train", "train-evidence", "final-eval", "final-evidence"):
        recipe = makefile.split(f"aws-rl-harbor-generalization-{name}:\n", 1)[1].split("\n\n", 1)[0]
        assert "$(AWS_RUN)" in recipe
        assert "EXPECTED_AWS_ACCOUNT_ID=" not in recipe


def test_chat_defaults_and_explicit_overrides():
    for target, expected in (
        ("chat", "scripted-correct"),
        ("chat-cli", "scripted-correct"),
        ("aws-chat", "bedrock"),
    ):
        result = subprocess.run(["make", "-n", target], check=True, text=True, capture_output=True)
        assert f'--candidate "{expected}"' in result.stdout
        result = subprocess.run(
            ["make", "-n", target, "AGENTCORE_CANDIDATE=scripted-bad"],
            check=True,
            text=True,
            capture_output=True,
        )
        assert '--candidate "scripted-bad"' in result.stdout
