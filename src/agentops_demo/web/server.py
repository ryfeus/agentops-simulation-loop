"""Serve the checkpointed billing LangGraph over the official AG-UI adapter."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentops_demo.agent.local import (
    local_agent_config_from_environment,
    local_candidate_from_environment,
)
from agentops_demo.agent.runtime import AgentSessionRuntime, create_session_runtime
from agentops_demo.agentcore.request import CandidateSelector
from agentops_demo.contracts.agent_config import AgentConfig

LOGGER = logging.getLogger(__name__)
ALLOWED_ORIGINS = ("http://127.0.0.1:3000", "http://localhost:3000")


@dataclass(frozen=True)
class WebServerSettings:
    """Configuration for one local browser-chat backend process."""

    candidate: CandidateSelector
    config: AgentConfig
    mcp_url: str

    @classmethod
    def from_environment(cls, candidate: str | None = None) -> WebServerSettings:
        selected = local_candidate_from_environment(candidate)
        return cls(
            candidate=selected,
            config=local_agent_config_from_environment(selected),
            mcp_url=os.getenv("BILLING_MCP_URL", "http://127.0.0.1:8000/mcp"),
        )


RuntimeFactory = Callable[..., Awaitable[AgentSessionRuntime]]


def create_web_app(
    settings: WebServerSettings,
    *,
    runtime_factory: RuntimeFactory = create_session_runtime,
) -> FastAPI:
    """Create the local-only AG-UI application.

    The graph is deliberately bootstrapped once inside the server lifespan.  The
    official adapter maps each AG-UI ``threadId`` to LangGraph's thread ID, so
    its checkpointer remains local to this process and page-scoped in practice.
    """

    runtime_box: dict[str, AgentSessionRuntime] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            runtime = await runtime_factory(config=settings.config, mcp_url=settings.mcp_url)
            if not isinstance(runtime, AgentSessionRuntime):
                raise RuntimeError("local agent runtime factory returned an invalid runtime")
            runtime_box["runtime"] = runtime
            agent = LangGraphAgent(name="billing-agent", graph=runtime.graph)
            add_langgraph_fastapi_endpoint(app, agent, path="/agent")
        except Exception as exc:
            LOGGER.exception("local AG-UI runtime bootstrap failed")
            raise RuntimeError("Unable to start the local agent runtime") from exc
        yield

    app = FastAPI(title="AgentOps local AG-UI", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(ALLOWED_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok" if "runtime" in runtime_box else "starting",
            "candidate": settings.candidate,
            "model": f"{settings.config.model.provider}/{settings.config.model.model_id}",
            "agent_config_fingerprint": settings.config.fingerprint(),
            "source_revision": settings.config.agent.source_revision,
        }

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local AG-UI billing backend")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--host", default=os.getenv("AGUI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AGUI_PORT", "8080")))
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    settings = WebServerSettings.from_environment(args.candidate)
    import uvicorn

    uvicorn.run(create_web_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
