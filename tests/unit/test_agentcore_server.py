from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from agentops_demo.agentcore import server as server_module
from agentops_demo.agentcore.candidates import resolve_candidate
from agentops_demo.agentcore.request import AgentCoreRequest
from agentops_demo.agentcore.server import InvocationTracker, create_agentcore_app
from agentops_demo.agentcore.settings import AgentCoreSettings


class FakeManager:
    def __init__(self) -> None:
        self.ready = False
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True
        self.ready = True

    async def stop(self) -> None:
        self.ready = False
        self.stopped = True


def test_agentcore_http_contract(agent_config) -> None:
    manager = FakeManager()

    async def invoke(**_kwargs: Any) -> dict[str, Any]:
        return {
            "final_response": "Invoice inv-123 is disputed.",
            "tool_calls": [{"name": "get_invoice", "arguments": {"invoice_id": "inv-123"}}],
            "completed_tools": ["get_invoice"],
            "agent_config_fingerprint": agent_config.fingerprint(),
        }

    app = create_agentcore_app(
        AgentCoreSettings(config=agent_config),
        invocation=invoke,
        process_manager=manager,  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        ping = client.get("/ping")
        assert ping.status_code == 200
        assert ping.json()["status"] == "Healthy"
        response = client.post(
            "/invocations",
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "session-123"},
            json={"instruction": "Inspect inv-123"},
        )
        assert response.status_code == 200
        assert response.json()["completed_tools"] == ["get_invoice"]
        assert response.json()["source_revision"] == "abc123"
    assert manager.started
    assert manager.stopped


def test_agentcore_request_is_strict(agent_config) -> None:
    manager = FakeManager()

    async def invoke(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("invalid requests must not invoke")

    app = create_agentcore_app(
        AgentCoreSettings(config=agent_config),
        invocation=invoke,
        process_manager=manager,  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        assert client.post("/invocations", json={"instruction": "   "}).status_code == 422
        assert (
            client.post("/invocations", json={"instruction": "ok", "extra": True}).status_code
            == 422
        )
        assert (
            client.post(
                "/invocations",
                headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "invalid"},
                json={"instruction": "ok", "candidate": "unknown"},
            ).status_code
            == 422
        )


@pytest.mark.parametrize("candidate", (None, "bedrock", "scripted-bad", "scripted-correct"))
def test_agentcore_request_accepts_only_supported_candidates(candidate: str | None) -> None:
    request = AgentCoreRequest.model_validate({"instruction": "ok", "candidate": candidate})
    assert request.candidate == candidate


def test_agentcore_request_rejects_unknown_candidate() -> None:
    with pytest.raises(ValueError):
        AgentCoreRequest.model_validate({"instruction": "ok", "candidate": "unknown"})


def test_agentcore_masks_invocation_failures(agent_config) -> None:
    manager = FakeManager()

    async def invoke(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("secret detail")

    app = create_agentcore_app(
        AgentCoreSettings(config=agent_config),
        invocation=invoke,
        process_manager=manager,  # type: ignore[arg-type]
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/invocations",
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "failure-session"},
            json={"instruction": "inspect"},
        )
        assert client.get("/ping").json()["status"] == "Healthy"
    assert response.status_code == 500
    assert response.json() == {"detail": "invocation_failed"}
    assert "secret detail" not in response.text


def test_agentcore_requires_a_nonblank_runtime_session_id(agent_config) -> None:
    manager = FakeManager()
    calls: list[dict[str, Any]] = []

    async def invoke(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return successful_result(agent_config)

    app = create_agentcore_app(
        AgentCoreSettings(config=agent_config),
        invocation=invoke,
        process_manager=manager,  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        assert client.post("/invocations", json={"instruction": "inspect"}).json() == {
            "detail": "runtime_session_id_required"
        }
        assert (
            client.post(
                "/invocations",
                headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "   "},
                json={"instruction": "inspect"},
            ).status_code
            == 400
        )
    assert calls == []


def test_agentcore_propagates_runtime_session_id_to_invocation(agent_config) -> None:
    manager = FakeManager()
    received: dict[str, Any] = {}

    async def invoke(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return successful_result(agent_config)

    app = create_agentcore_app(
        AgentCoreSettings(config=agent_config),
        invocation=invoke,
        process_manager=manager,  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        response = client.post(
            "/invocations",
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "exact-session-id"},
            json={"instruction": "inspect"},
        )
    assert response.status_code == 200
    assert received["session_id"] == "exact-session-id"


def test_agentcore_uses_effective_candidate_config_and_binds_the_session(
    agent_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = FakeManager()
    seen: list[tuple[str, str]] = []
    logs: list[dict[str, object]] = []

    monkeypatch.setattr(
        server_module,
        "_log",
        lambda event, **fields: logs.append({"event": event, **fields}),
    )

    async def invoke(*, config, **_kwargs: Any) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        seen.append((config.model.provider, config.model.model_id))
        return {
            **successful_result(config),
            "agent_config_fingerprint": config.fingerprint(),
        }

    app = create_agentcore_app(
        AgentCoreSettings(config=agent_config),
        invocation=invoke,
        process_manager=manager,  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        bad = client.post(
            "/invocations",
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "bad-session"},
            json={"instruction": "inspect", "candidate": "scripted-bad"},
        )
        same = client.post(
            "/invocations",
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "bad-session"},
            json={"instruction": "inspect", "candidate": "scripted-bad"},
        )
        conflict = client.post(
            "/invocations",
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "bad-session"},
            json={"instruction": "inspect", "candidate": "scripted-correct"},
        )
        correct = client.post(
            "/invocations",
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "correct-session"},
            json={"instruction": "inspect", "candidate": "scripted-correct"},
        )
    assert bad.status_code == same.status_code == correct.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "runtime_session_candidate_conflict"}
    assert bad.json()["agent_config_fingerprint"] != agent_config.fingerprint()
    assert seen == [("scripted", "bad"), ("scripted", "bad"), ("scripted", "correct")]
    started = [entry for entry in logs if entry["event"] == "invocation_started"]
    assert started[0]["candidate"] == "scripted-bad"
    assert started[0]["model"] == "scripted/bad"


def test_agentcore_propagates_runtime_session_id_to_tracing(
    agent_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = FakeManager()
    traced: dict[str, Any] = {}

    @asynccontextmanager
    async def trace_context(**kwargs: Any):
        traced.update(kwargs)
        yield

    async def invoke(**_kwargs: Any) -> dict[str, Any]:
        return successful_result(agent_config)

    monkeypatch.setattr(server_module, "invocation_trace_context", trace_context)
    app = create_agentcore_app(
        AgentCoreSettings(config=agent_config),
        invocation=invoke,
        process_manager=manager,  # type: ignore[arg-type]
    )
    effective = resolve_candidate(agent_config, "scripted-bad")
    with TestClient(app) as client:
        response = client.post(
            "/invocations",
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "traced-session"},
            json={"instruction": "inspect", "candidate": "scripted-bad"},
        )
    assert response.status_code == 200
    assert traced["session_id"] == "traced-session"
    assert traced["config"].fingerprint() == effective.fingerprint()


def successful_result(agent_config) -> dict[str, Any]:
    return {
        "final_response": "done",
        "tool_calls": [],
        "completed_tools": [],
        "agent_config_fingerprint": agent_config.fingerprint(),
    }


@pytest.mark.asyncio
async def test_ping_is_busy_during_active_invocation(agent_config) -> None:
    manager = FakeManager()
    manager.ready = True
    started = asyncio.Event()
    release = asyncio.Event()

    async def invoke(**_kwargs: Any) -> dict[str, Any]:
        started.set()
        await release.wait()
        return successful_result(agent_config)

    app = create_agentcore_app(
        AgentCoreSettings(config=agent_config),
        invocation=invoke,
        process_manager=manager,  # type: ignore[arg-type]
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        request = asyncio.create_task(
            client.post(
                "/invocations",
                headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "busy-session"},
                json={"instruction": "inspect"},
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        busy = await client.get("/ping")
        assert busy.status_code == 200
        assert busy.json()["status"] == "HealthyBusy"
        release.set()
        assert (await request).status_code == 200
        assert (await client.get("/ping")).json()["status"] == "Healthy"


@pytest.mark.asyncio
async def test_ping_remains_busy_until_all_concurrent_invocations_finish(agent_config) -> None:
    manager = FakeManager()
    manager.ready = True
    starts = {name: asyncio.Event() for name in ("one", "two")}
    releases = {name: asyncio.Event() for name in ("one", "two")}

    async def invoke(*, instruction: str, **_kwargs: Any) -> dict[str, Any]:
        starts[instruction].set()
        await releases[instruction].wait()
        return successful_result(agent_config)

    app = create_agentcore_app(
        AgentCoreSettings(config=agent_config),
        invocation=invoke,
        process_manager=manager,  # type: ignore[arg-type]
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        requests = {
            name: asyncio.create_task(
                client.post(
                    "/invocations",
                    headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": f"session-{name}"},
                    json={"instruction": name},
                )
            )
            for name in ("one", "two")
        }
        waits = (asyncio.wait_for(event.wait(), timeout=1) for event in starts.values())
        await asyncio.gather(*waits)
        assert (await client.get("/ping")).json()["status"] == "HealthyBusy"
        releases["one"].set()
        assert (await requests["one"]).status_code == 200
        assert (await client.get("/ping")).json()["status"] == "HealthyBusy"
        releases["two"].set()
        assert (await requests["two"]).status_code == 200
        assert (await client.get("/ping")).json()["status"] == "Healthy"


@pytest.mark.asyncio
async def test_cancelled_invocation_releases_busy_state(agent_config) -> None:
    manager = FakeManager()
    manager.ready = True
    tracker = InvocationTracker()
    started = asyncio.Event()

    async def invoke(**_kwargs: Any) -> dict[str, Any]:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    app = create_agentcore_app(
        AgentCoreSettings(config=agent_config),
        invocation=invoke,
        process_manager=manager,  # type: ignore[arg-type]
        invocation_tracker=tracker,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        request = asyncio.create_task(
            client.post(
                "/invocations",
                headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "cancelled-session"},
                json={"instruction": "inspect"},
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        assert await tracker.active_count() == 1
        request.cancel()
        with suppress(asyncio.CancelledError):
            await request
        assert await tracker.active_count() == 0
        assert (await client.get("/ping")).json()["status"] == "Healthy"


@pytest.mark.asyncio
async def test_ping_is_unhealthy_when_mcp_is_unavailable(agent_config) -> None:
    manager = FakeManager()
    app = create_agentcore_app(
        AgentCoreSettings(config=agent_config),
        process_manager=manager,  # type: ignore[arg-type]
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ping")
    assert response.status_code == 503
    assert response.json() == {"status": "Unhealthy"}
