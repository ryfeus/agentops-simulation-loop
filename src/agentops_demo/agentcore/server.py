"""FastAPI adapter implementing the AgentCore custom-container HTTP contract."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from agentops_demo.agentcore.candidates import (
    CandidateResolutionError,
    CandidateRuntimePool,
    SessionCandidateBindings,
    SessionCandidateConflictError,
    resolve_candidate,
)
from agentops_demo.agentcore.request import AgentCoreRequest, AgentCoreResponse
from agentops_demo.agentcore.settings import AgentCoreSettings
from agentops_demo.agentcore.tracing import invocation_trace_context, world_snapshot_attributes
from agentops_demo.billing.repository_factory import create_billing_world_inspector
from agentops_demo.cli.run_once import AgentSessionRuntime, create_session_runtime
from agentops_demo.contracts.billing import BillingWorldInspector

Invocation = Callable[..., Awaitable[dict[str, Any]]]
SessionRuntimeFactory = Callable[..., Awaitable[AgentSessionRuntime]]


def _log(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


class MCPProcessManager:
    """Own the co-located Billing MCP process for one AgentCore container."""

    def __init__(self, settings: AgentCoreSettings) -> None:
        self.settings = settings
        self.process: subprocess.Popen[bytes] | None = None
        self.ready = False

    async def start(self) -> None:
        environment = {
            **os.environ,
            "BILLING_MCP_HOST": self.settings.mcp_host,
            "BILLING_MCP_PORT": str(self.settings.mcp_port),
            "PYTHONUNBUFFERED": "1",
        }
        self.process = subprocess.Popen(
            [sys.executable, "-m", "agentops_demo.mcp.billing_server"],
            env=environment,
        )
        await self._wait_until_ready()
        self.ready = True
        _log("mcp_started", host=self.settings.mcp_host, port=self.settings.mcp_port)

    async def _wait_until_ready(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.mcp_startup_timeout
        last_error = "server did not accept connections"
        while loop.time() < deadline:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("Billing MCP exited during AgentCore startup")
            try:
                reader, writer = await asyncio.open_connection(
                    self.settings.mcp_host, self.settings.mcp_port
                )
                writer.write(
                    (
                        "GET /health HTTP/1.1\r\n"
                        f"Host: {self.settings.mcp_host}:{self.settings.mcp_port}\r\n"
                        "Connection: close\r\n\r\n"
                    ).encode()
                )
                await writer.drain()
                response = await asyncio.wait_for(reader.read(1024), timeout=1)
                writer.close()
                await writer.wait_closed()
                if b"200 OK" in response:
                    return
                last_error = response.decode(errors="replace")
            except (OSError, TimeoutError) as exc:
                last_error = str(exc)
            await asyncio.sleep(0.05)
        raise RuntimeError(
            f"Billing MCP was not ready after {self.settings.mcp_startup_timeout}s: {last_error}"
        )

    async def stop(self) -> None:
        self.ready = False
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            await asyncio.to_thread(self.process.wait, 5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            await asyncio.to_thread(self.process.wait, 5)
        _log("mcp_stopped")


class InvocationTracker:
    """Track process-local in-flight invocations for AgentCore health responses."""

    def __init__(self) -> None:
        self._active = 0
        self._lock = asyncio.Lock()

    async def active_count(self) -> int:
        async with self._lock:
            return self._active

    @asynccontextmanager
    async def invocation(self) -> AsyncIterator[None]:
        async with self._lock:
            self._active += 1
        try:
            yield
        finally:
            async with self._lock:
                self._active -= 1


def create_agentcore_app(
    settings: AgentCoreSettings,
    *,
    invocation: Invocation | None = None,
    session_runtime_factory: SessionRuntimeFactory = create_session_runtime,
    process_manager: MCPProcessManager | None = None,
    invocation_tracker: InvocationTracker | None = None,
    world_inspector: BillingWorldInspector | None = None,
) -> FastAPI:
    """Create the HTTP adapter with injectable session runtime and MCP lifecycle."""

    manager = process_manager or MCPProcessManager(settings)
    tracker = invocation_tracker or InvocationTracker()
    inspector = world_inspector
    if inspector is None and settings.world_snapshot_enabled:
        inspector = create_billing_world_inspector(
            backend=os.getenv("BILLING_REPOSITORY_BACKEND", "sqlite"),
            sqlite_path=os.getenv("BILLING_DATABASE_PATH", "data/billing.db"),
        )
    default_fingerprint = settings.config.fingerprint()
    revision = settings.config.agent.source_revision
    runtime_pool = CandidateRuntimePool(
        mcp_url=settings.mcp_url,
        runtime_factory=session_runtime_factory,
    )
    session_candidates = SessionCandidateBindings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        _log(
            "runtime_boot",
            default_agent_config_fingerprint=default_fingerprint,
            source_revision=revision,
            default_model=f"{settings.config.model.provider}/{settings.config.model.model_id}",
        )
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/ping")
    async def ping() -> JSONResponse:
        if not manager.ready:
            return JSONResponse(status_code=503, content={"status": "Unhealthy"})
        status = "HealthyBusy" if await tracker.active_count() else "Healthy"
        return JSONResponse(
            content={
                "status": status,
                "agent_config_fingerprint": default_fingerprint,
                "source_revision": revision,
            }
        )

    @app.post("/invocations", response_model=AgentCoreResponse)
    async def invocations(payload: AgentCoreRequest, request: Request) -> AgentCoreResponse:
        async with tracker.invocation():
            session_id = request.headers.get("x-amzn-bedrock-agentcore-runtime-session-id")
            if session_id is None or not session_id.strip():
                raise HTTPException(status_code=400, detail="runtime_session_id_required")
            try:
                config = resolve_candidate(settings.config, payload.candidate)
            except CandidateResolutionError:
                raise HTTPException(status_code=400, detail="candidate_unavailable") from None
            fingerprint = config.fingerprint()
            selector = payload.candidate or "default"
            try:
                runtime = None if invocation is not None else await runtime_pool.runtime_for(config)
                await session_candidates.bind_or_validate(
                    session_id=session_id,
                    fingerprint=fingerprint,
                )
            except SessionCandidateConflictError:
                raise HTTPException(
                    status_code=409,
                    detail="runtime_session_candidate_conflict",
                ) from None
            except Exception:
                _log(
                    "candidate_runtime_failed",
                    runtime_session_id=session_id,
                    candidate=selector,
                    model=f"{config.model.provider}/{config.model.model_id}",
                    agent_config_fingerprint=fingerprint,
                    source_revision=config.agent.source_revision,
                )
                raise HTTPException(status_code=500, detail="invocation_failed") from None
            snapshot = None
            if settings.world_snapshot_enabled:
                assert inspector is not None
                try:
                    snapshot = await inspector.snapshot_world()
                    # Validate size and serialization before entering the trace context so an
                    # optional capture failure cannot accidentally re-run the invocation.
                    world_snapshot_attributes(
                        snapshot, maximum_bytes=settings.world_snapshot_max_bytes
                    )
                except Exception as exc:
                    _log("world_snapshot_failed", runtime_session_id=session_id, error=str(exc))
                    snapshot = None
                    if settings.world_snapshot_required:
                        raise HTTPException(status_code=500, detail="invocation_failed") from None
            async with invocation_trace_context(
                config=config,
                policy_mode=os.getenv("BILLING_POLICY_MODE", "permissive"),
                session_id=session_id,
                instruction=payload.instruction if settings.trace_content_enabled else None,
                world_snapshot=snapshot,
                world_snapshot_max_bytes=settings.world_snapshot_max_bytes,
                benchmark_context=payload.evaluation_context,
            ):
                started = time.monotonic()
                _log(
                    "invocation_started",
                    runtime_session_id=session_id,
                    candidate=selector,
                    model=f"{config.model.provider}/{config.model.model_id}",
                    agent_config_fingerprint=fingerprint,
                    source_revision=config.agent.source_revision,
                )
                try:
                    if invocation is not None:
                        result = await invocation(
                            config=config,
                            instruction=payload.instruction,
                            mcp_url=settings.mcp_url,
                            session_id=session_id,
                        )
                    else:
                        if runtime is None:
                            raise RuntimeError("AgentCore session runtime is not initialized")
                        result = await runtime.invoke(
                            session_id=session_id,
                            instruction=payload.instruction,
                        )
                    response = AgentCoreResponse.model_validate(
                        {**result, "source_revision": revision}
                    )
                except Exception:
                    _log(
                        "invocation_failed",
                        runtime_session_id=session_id,
                        candidate=selector,
                        model=f"{config.model.provider}/{config.model.model_id}",
                        agent_config_fingerprint=fingerprint,
                        source_revision=config.agent.source_revision,
                        duration_ms=round((time.monotonic() - started) * 1000),
                    )
                    raise HTTPException(status_code=500, detail="invocation_failed") from None
            _log(
                "invocation_completed",
                runtime_session_id=session_id,
                candidate=selector,
                model=f"{config.model.provider}/{config.model.model_id}",
                agent_config_fingerprint=fingerprint,
                source_revision=config.agent.source_revision,
                tool_names=response.completed_tools,
                duration_ms=round((time.monotonic() - started) * 1000),
            )
            return response

    return app


def create_app_from_environment() -> FastAPI:
    return create_agentcore_app(AgentCoreSettings.from_environment())
