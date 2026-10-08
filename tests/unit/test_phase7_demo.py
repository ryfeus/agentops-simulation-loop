"""Credential-free integrity checks for Phase 7 evidence staging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_e2e_demo


def test_completed_stage_binds_named_artifact_hashes(tmp_path: Path) -> None:
    artifact = tmp_path / "production" / "failure.json"
    artifact.parent.mkdir()
    artifact.write_text('{"failure": true}\n')
    run_e2e_demo._stage(
        tmp_path,
        2,
        "production_failure",
        "2026-09-11T00:00:00+00:00",
        {"failure": artifact},
    )
    assert run_e2e_demo._completed(tmp_path, 2, "production_failure")
    artifact.write_text('{"failure": false}\n')
    with pytest.raises(RuntimeError, match="missing or altered"):
        run_e2e_demo._completed(tmp_path, 2, "production_failure")


def test_resume_rejects_git_revision_change_before_reusing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "run.json").write_text(json.dumps({"git_revision": "old"}))
    monkeypatch.setattr(run_e2e_demo, "current_revision", lambda: "new")
    with pytest.raises(RuntimeError, match="different Git revision"):
        run_e2e_demo._validate_resume(tmp_path)


def test_replay_requires_explicit_source_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", "123456789012")
    monkeypatch.setattr(run_e2e_demo, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="requires --source-run"):
        run_e2e_demo.run(run_id="replay", mode="replay")


def test_replay_rejects_candidate_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", "123456789012")
    monkeypatch.setattr(run_e2e_demo, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="does not accept a candidate override"):
        run_e2e_demo.run(
            run_id="replay",
            mode="replay",
            candidate="scripted-bad",
            source_run=tmp_path / "source",
        )


@pytest.mark.parametrize("candidate", [None, "scripted-bad"])
def test_live_preflight_receives_default_or_explicit_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate: str | None
) -> None:
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", "123456789012")
    monkeypatch.setattr(run_e2e_demo, "ROOT", tmp_path)
    seen: list[str | None] = []

    def stop_after_preflight(_run_id: str, *, candidate: str | None) -> None:
        seen.append(candidate)
        raise RuntimeError("stop after preflight")

    monkeypatch.setattr(run_e2e_demo, "preflight", stop_after_preflight)
    with pytest.raises(RuntimeError, match="stop after preflight"):
        run_e2e_demo.run(run_id="live", mode="live", candidate=candidate)
    assert seen == [candidate or "bedrock"]
