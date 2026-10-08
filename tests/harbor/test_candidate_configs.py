from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentops_demo.harbor import provenance
from scripts.generate_harbor_configs import require_clean_worktree, write_configs


def test_candidate_configs_differ_only_by_scripted_model(tmp_path: Path) -> None:
    configs = write_configs(tmp_path, "a" * 40)

    assert configs["bad"].agent.source_revision == "a" * 40
    assert configs["correct"].agent.source_revision == "a" * 40
    assert configs["noop"].agent.source_revision == "a" * 40
    assert configs["bad"].prompt == configs["correct"].prompt
    assert configs["bad"].tools == configs["correct"].tools
    assert configs["bad"].harness == configs["correct"].harness
    assert configs["bad"].model.model_id == "bad"
    assert configs["correct"].model.model_id == "correct"
    assert configs["noop"].model.model_id == "noop"
    assert (tmp_path / "bad.json").is_file()
    assert (tmp_path / "correct.json").is_file()
    assert (tmp_path / "noop.json").is_file()


def test_clean_worktree_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def clean_run(command: tuple[str, ...], **_: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(provenance.subprocess, "run", clean_run)

    require_clean_worktree()

    assert commands == [("git", "status", "--porcelain", "--untracked-files=all")]


@pytest.mark.parametrize(
    "status",
    [
        " M src/agentops_demo/changed.py\n",
        "M  src/agentops_demo/staged.py\n",
        " D benchmarks/deleted.txt\n",
        "R  old.py -> new.py\n",
        "?? src/agentops_demo/untracked.py\n",
        "?? benchmarks/disputed-refund/untracked.txt\n",
    ],
)
def test_dirty_worktree_entries_are_rejected(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    def dirty_run(command: tuple[str, ...], **_: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=status)

    monkeypatch.setattr(provenance.subprocess, "run", dirty_run)

    with pytest.raises(RuntimeError, match="working tree must be clean"):
        require_clean_worktree()


def test_ignored_output_not_surfaced_by_git_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ignored_run(command: tuple[str, ...], **_: object) -> SimpleNamespace:
        assert command == ("git", "status", "--porcelain", "--untracked-files=all")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(provenance.subprocess, "run", ignored_run)

    require_clean_worktree()


def test_untracked_plan_document_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def plan_run(command: tuple[str, ...], **_: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="?? docs/untracked-example.md\n")

    monkeypatch.setattr(provenance.subprocess, "run", plan_run)
    with pytest.raises(RuntimeError, match="working tree must be clean"):
        require_clean_worktree()
