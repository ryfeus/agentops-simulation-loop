"""Supervise the local MCP, AG-UI, and Vite processes for ``make chat``."""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import time
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol
from urllib.error import URLError
from urllib.request import urlopen

from agentops_demo.agent.local import local_candidate_from_environment
from agentops_demo.cli.init_db import DEFAULT_SCENARIO, initialize_from_scenario

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "data" / "billing.db"
MCP_HEALTH_URL = "http://127.0.0.1:8000/health"
AGUI_HEALTH_URL = "http://127.0.0.1:8080/health"
CHAT_URL = "http://127.0.0.1:3000"


class Process(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


PopenFactory = Callable[..., Process]
HealthProbe = Callable[[str], bool]
PortProbe = Callable[[str, int], bool]


def database_path_from_environment() -> Path:
    """Return the local SQLite path, resolved relative to the repository."""

    value = Path(os.getenv("BILLING_DATABASE_PATH", str(DEFAULT_DATABASE)))
    return value if value.is_absolute() else ROOT / value


def ensure_database(database_path: Path) -> bool:
    """Create the scenario database only when it is absent.

    ``initialize_database`` is intentionally destructive, so this guard is the
    contract that preserves an existing local chat's billing state.
    """

    if database_path.exists():
        return False
    database_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(initialize_from_scenario(DEFAULT_SCENARIO, database_path))
    return True


def http_is_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=0.5) as response:
            return 200 <= response.status < 300
    except (OSError, URLError):
        return False


def port_is_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def wait_for(
    *,
    name: str,
    ready: Callable[[], bool],
    process: Process,
    timeout_seconds: float = 20,
) -> None:
    """Wait for one child while surfacing early process exits clearly."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = process.poll()
        if status is not None:
            raise RuntimeError(f"{name} exited during startup (status {status})")
        if ready():
            return
        time.sleep(0.1)
    raise RuntimeError(f"{name} did not become ready within {timeout_seconds:g} seconds")


def terminate_all(processes: Sequence[Process]) -> None:
    """Terminate and reap children in reverse dependency order."""

    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    for process in reversed(processes):
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def child_environment(*, candidate: str, database_path: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "AGENTCORE_CANDIDATE": candidate,
            "BILLING_DATABASE_PATH": str(database_path),
            "BILLING_MCP_HOST": "127.0.0.1",
            "BILLING_MCP_PORT": "8000",
            "BILLING_MCP_URL": "http://127.0.0.1:8000/mcp",
            "AGUI_HOST": "127.0.0.1",
            "AGUI_PORT": "8080",
        }
    )
    return environment


def run_stack(
    *,
    candidate: str,
    database_path: Path,
    popen: PopenFactory = subprocess.Popen,
    health_probe: HealthProbe = http_is_ready,
    port_probe: PortProbe = port_is_ready,
    browser_open: Callable[[str], bool] = webbrowser.open,
) -> None:
    """Launch the browser chat stack and remain attached until interrupted."""

    environment = child_environment(candidate=candidate, database_path=database_path)
    children: list[Process] = []
    try:
        mcp = popen(
            [sys.executable, "-m", "agentops_demo.mcp.billing_server"],
            cwd=ROOT,
            env=environment,
        )
        children.append(mcp)
        wait_for(
            name="MCP server",
            ready=lambda: health_probe(MCP_HEALTH_URL),
            process=mcp,
        )

        agui = popen(
            [sys.executable, "-m", "agentops_demo.web.server", "--candidate", candidate],
            cwd=ROOT,
            env=environment,
        )
        children.append(agui)
        wait_for(
            name="AG-UI backend",
            ready=lambda: health_probe(AGUI_HEALTH_URL),
            process=agui,
        )

        vite = popen(
            ["pnpm", "--dir", "web", "dev", "--host", "127.0.0.1", "--port", "3000"],
            cwd=ROOT,
            env=environment,
        )
        children.append(vite)
        wait_for(
            name="Vite frontend",
            ready=lambda: port_probe("127.0.0.1", 3000),
            process=vite,
        )
        browser_open(CHAT_URL)
        print(f"Local browser chat is ready at {CHAT_URL} ({candidate}). Press Ctrl-C to stop.")

        while True:
            for name, process in zip(
                ("MCP server", "AG-UI backend", "Vite frontend"), children, strict=True
            ):
                status = process.poll()
                if status is not None:
                    raise RuntimeError(f"{name} exited (status {status})")
            time.sleep(0.25)
    finally:
        terminate_all(children)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local browser billing chat")
    parser.add_argument("--candidate", default=None)
    args = parser.parse_args(argv)
    candidate = local_candidate_from_environment(args.candidate)
    database_path = database_path_from_environment()
    created = ensure_database(database_path)
    if created:
        print(f"Initialized local SQLite database at {database_path}")
    else:
        print(f"Using existing local SQLite database at {database_path}")
    try:
        run_stack(candidate=candidate, database_path=database_path)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
