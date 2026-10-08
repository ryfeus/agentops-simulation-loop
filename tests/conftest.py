from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from agentops_demo.billing.sqlite_repository import initialize_database
from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.validation.scenario import load_scenario

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SCENARIO = REPOSITORY_ROOT / "scenarios" / "disputed-refund" / "scenario.yaml"
sys.path.insert(0, str(REPOSITORY_ROOT))


@pytest.fixture
def agent_config_data() -> dict[str, object]:
    return {
        "agent": {"source_revision": "abc123"},
        "model": {"provider": "bedrock", "model_id": "model-v1"},
        "prompt": {"version": "billing-v1"},
        "tools": {"version": "billing-mcp-v1"},
        "harness": {"framework": "langgraph", "version": "v1"},
    }


@pytest.fixture
def agent_config(agent_config_data: dict[str, object]) -> AgentConfig:
    return AgentConfig.model_validate(agent_config_data)


@dataclass
class RunningMCPServer:
    url: str
    database_path: Path
    process: subprocess.Popen[str]

    async def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            await asyncio.to_thread(self.process.wait, 5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            await asyncio.to_thread(self.process.wait, 5)


MCPServerFactory = Callable[[str], Awaitable[RunningMCPServer]]


@pytest.fixture
async def mcp_server_factory(tmp_path: Path) -> MCPServerFactory:
    servers: list[RunningMCPServer] = []

    async def start(policy_mode: str = "permissive") -> RunningMCPServer:
        database_path = tmp_path / f"billing-{len(servers)}.db"
        scenario = load_scenario(GOLDEN_SCENARIO)
        await initialize_database(database_path, scenario.initial_state)
        host = "127.0.0.1"
        port = _available_port(host)
        environment = {
            **os.environ,
            "BILLING_DATABASE_PATH": str(database_path),
            "BILLING_POLICY_MODE": policy_mode,
            "BILLING_MCP_HOST": host,
            "BILLING_MCP_PORT": str(port),
            "PYTHONUNBUFFERED": "1",
        }
        process = subprocess.Popen(
            [sys.executable, "-m", "agentops_demo.mcp.billing_server"],
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        server = RunningMCPServer(
            url=f"http://{host}:{port}/mcp",
            database_path=database_path,
            process=process,
        )
        servers.append(server)
        await _wait_until_ready(server, host, port)
        return server

    try:
        yield start
    finally:
        for server in reversed(servers):
            await server.stop()
            assert server.process.poll() is not None


def _available_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


async def _wait_until_ready(
    server: RunningMCPServer,
    host: str,
    port: int,
    *,
    timeout: float = 15,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last_error = "server did not accept connections"
    while loop.time() < deadline:
        if server.process.poll() is not None:
            stderr = server.process.stderr.read() if server.process.stderr else ""
            raise RuntimeError(f"MCP server exited during startup: {stderr}")
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(
                f"GET /health HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n".encode()
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
    raise RuntimeError(f"MCP server was not ready after {timeout}s: {last_error}")
