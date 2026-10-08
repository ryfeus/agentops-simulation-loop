from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from scripts import invoke_agentcore


@pytest.fixture(autouse=True)
def aws_identity(monkeypatch):
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", "123456789012")


class Body:
    def read(self) -> bytes:
        return json.dumps(
            {
                "final_response": "disputed",
                "tool_calls": [],
                "completed_tools": ["get_invoice"],
            }
        ).encode()


def test_invoke_agentcore_reads_streaming_response(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class Client:
        def invoke_agent_runtime(self, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {"response": Body(), "runtimeSessionId": kwargs["runtimeSessionId"]}

    monkeypatch.setattr(
        invoke_agentcore,
        "terraform_outputs",
        lambda: {
            "aws_region": "us-west-2",
            "agentcore_runtime_arn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/x",
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(
            Session=lambda **kwargs: SimpleNamespace(
                client=lambda service, **kwargs: (
                    SimpleNamespace(get_caller_identity=lambda: {"Account": "123456789012"})
                    if service == "sts"
                    else Client()
                )
            )
        ),
    )

    result, session_id = invoke_agentcore.invoke(
        "Inspect invoice inv-123", session_id="requested-session"
    )
    assert session_id == "requested-session"
    assert result["completed_tools"] == ["get_invoice"]
    assert calls[0]["qualifier"] == "DEFAULT"
    assert calls[0]["runtimeSessionId"] == "requested-session"
    assert json.loads(calls[0]["payload"]) == {"instruction": "Inspect invoice inv-123"}


def test_invoke_agentcore_includes_candidate_only_when_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class Client:
        def invoke_agent_runtime(self, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {"response": Body(), "runtimeSessionId": kwargs["runtimeSessionId"]}

    monkeypatch.setattr(
        invoke_agentcore,
        "terraform_outputs",
        lambda: {
            "aws_region": "us-west-2",
            "agentcore_runtime_arn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/x",
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(
            Session=lambda **kwargs: SimpleNamespace(
                client=lambda service, **kwargs: (
                    SimpleNamespace(get_caller_identity=lambda: {"Account": "123456789012"})
                    if service == "sts"
                    else Client()
                )
            )
        ),
    )

    invoke_agentcore.invoke("inspect", session_id="without-candidate")
    invoke_agentcore.invoke(
        "inspect",
        session_id="with-candidate",
        candidate="scripted-bad",
    )
    assert json.loads(calls[0]["payload"]) == {"instruction": "inspect"}
    assert json.loads(calls[1]["payload"]) == {
        "instruction": "inspect",
        "candidate": "scripted-bad",
    }


def test_invoke_agentcore_generates_a_session_when_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class Client:
        def invoke_agent_runtime(self, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {"response": Body(), "runtimeSessionId": kwargs["runtimeSessionId"]}

    monkeypatch.setattr(
        invoke_agentcore,
        "terraform_outputs",
        lambda: {
            "aws_region": "us-west-2",
            "agentcore_runtime_arn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/x",
        },
    )
    monkeypatch.setattr(invoke_agentcore.uuid, "uuid4", lambda: "fresh-session")
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(
            Session=lambda **kwargs: SimpleNamespace(
                client=lambda service, **kwargs: (
                    SimpleNamespace(get_caller_identity=lambda: {"Account": "123456789012"})
                    if service == "sts"
                    else Client()
                )
            )
        ),
    )

    _result, session_id = invoke_agentcore.invoke("Inspect invoice inv-123")
    assert session_id == "fresh-session"
    assert calls[0]["runtimeSessionId"] == "fresh-session"


def test_invoke_agentcore_rejects_a_mismatched_response_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        def invoke_agent_runtime(self, **_kwargs: object) -> dict[str, object]:
            return {"response": Body(), "runtimeSessionId": "different-session"}

    monkeypatch.setattr(
        invoke_agentcore,
        "terraform_outputs",
        lambda: {
            "aws_region": "us-west-2",
            "agentcore_runtime_arn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/x",
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(
            Session=lambda **kwargs: SimpleNamespace(
                client=lambda service, **kwargs: (
                    SimpleNamespace(get_caller_identity=lambda: {"Account": "123456789012"})
                    if service == "sts"
                    else Client()
                )
            )
        ),
    )

    with pytest.raises(RuntimeError, match="unexpected runtime session ID"):
        invoke_agentcore.invoke("Inspect invoice inv-123", session_id="requested-session")
