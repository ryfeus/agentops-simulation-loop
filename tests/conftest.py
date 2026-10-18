from __future__ import annotations

import asyncio
import hashlib
import json
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


@pytest.fixture(scope="session")
def authored_prior_root(tmp_path_factory):
    from agentops_demo.taskify.integrity import scenario_sha256
    from experiments.decision_model import focused_data, generalization_v3_holdout, next_data
    from rl.phase12.corpus import scenarios

    repository = REPOSITORY_ROOT
    root = tmp_path_factory.mktemp("synthetic-prior-corpora")
    for name in ("experiments", "benchmarks", "src", "rl"):
        (root / name).symlink_to(repository / name, target_is_directory=True)

    def corpus(relative, tasks):
        folder = root / relative
        folder.mkdir(parents=True)
        metadata = {}
        for task, split in tasks:
            path = folder / task.id
            path.mkdir()
            scenario = path / "scenario.yaml"
            scenario.write_text(json.dumps(task.model_dump(mode="json")))
            metadata[task.id] = {
                "task_path": str(path),
                "scenario_sha256": scenario_sha256(task),
                "scenario_raw_sha256": hashlib.sha256(scenario.read_bytes()).hexdigest(),
                "split": split,
            }
        (folder / "cohort-manifest.json").write_text(
            json.dumps(
                {
                    "tasks": metadata,
                    "split_task_counts": dict.fromkeys(
                        sorted({m["split"] for m in metadata.values()}), 1
                    ),
                }
            )
        )

    corpus(
        ".artifacts/decision-model/next-v1/corpus",
        [(task, split) for task, split, *_ in next_data.planned_tasks()],
    )
    corpus(
        ".artifacts/decision-model-focused/corpus",
        [(task, prototype["split"]) for prototype, task, _ in focused_data.plans()],
    )
    corpus(
        ".artifacts/decision-model-generalization-v3/corpus",
        [(task, "test") for task in generalization_v3_holdout.scenarios().values()],
    )
    # Synthetic acceptance flags test freeze's metadata validation. They are not
    # teacher trajectories and are never exported or claimed as observed results.
    demonstrations = root / next_data.data.SOURCE_DIR / "demonstrations/demonstrations.jsonl"
    demonstrations.parent.mkdir(parents=True)
    demonstrations.write_text(
        "".join(
            json.dumps({"task_id": task.id, "reward": 1, "complete": True, "truncated": False})
            + "\n"
            for task in scenarios()["train"].values()
        )
    )
    return root
