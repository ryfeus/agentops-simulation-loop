from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import run_web_chat


class FakeProcess:
    def __init__(self, statuses: list[int | None] | None = None) -> None:
        self.statuses = statuses or [None]
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.statuses.pop(0) if self.statuses else None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.killed:
            return -9
        return 0

    def kill(self) -> None:
        self.killed = True


def test_ensure_database_does_not_reset_existing_database(tmp_path: Path) -> None:
    database = tmp_path / "billing.db"
    database.write_text("existing state")

    assert run_web_chat.ensure_database(database) is False
    assert database.read_text() == "existing state"


def test_wait_for_reports_a_startup_exit() -> None:
    with pytest.raises(RuntimeError, match=r"MCP server exited during startup \(status 19\)"):
        run_web_chat.wait_for(
            name="MCP server",
            ready=lambda: False,
            process=FakeProcess([19]),
            timeout_seconds=0.1,
        )


def test_stack_reaps_all_started_children_when_an_essential_child_exits(tmp_path: Path) -> None:
    children = [FakeProcess(), FakeProcess(), FakeProcess()]
    commands: list[list[str]] = []

    def popen(command: list[str], **_kwargs: object) -> FakeProcess:
        commands.append(command)
        return children[len(commands) - 1]

    checks = iter([True, True])
    with pytest.raises(RuntimeError, match=r"MCP server exited \(status 7\)"):
        # The startup probes consume no process status; mark MCP failed when
        # the supervisor begins its first essential-child check.
        children[0].statuses = [None, 7]
        run_web_chat.run_stack(
            candidate="scripted-bad",
            database_path=tmp_path / "billing.db",
            popen=popen,
            health_probe=lambda _url: next(checks),
            port_probe=lambda _host, _port: True,
            browser_open=lambda _url: True,
        )

    assert len(commands) == 3
    assert all(child.terminated for child in children)


def test_terminate_all_kills_children_that_ignore_termination() -> None:
    class StubbornProcess(FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            if not self.killed:
                raise subprocess.TimeoutExpired("fake", timeout)
            return -9

    process = StubbornProcess()
    run_web_chat.terminate_all([process])
    assert process.terminated and process.killed
