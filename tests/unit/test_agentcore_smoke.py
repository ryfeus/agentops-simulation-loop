from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.assert_agentcore_smoke import assert_smoke


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value))
    return path


def valid_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    result = write(
        tmp_path / "result.json",
        {
            "final_response": "disputed",
            "tool_calls": [{"name": "get_invoice", "arguments": {}}],
            "completed_tools": ["get_invoice"],
            "agent_config_fingerprint": "f" * 64,
            "source_revision": "a" * 40,
        },
    )
    state = write(
        tmp_path / "state.json",
        {"invoice": {"status": "disputed"}, "refunds": 0, "escalations": 0},
    )
    manifest = write(
        tmp_path / "manifest.json",
        {
            "source_revision": "a" * 40,
            "agent_config": {"fingerprint": "f" * 64},
        },
    )
    return result, state, manifest


def test_valid_agentcore_smoke(tmp_path: Path) -> None:
    assert_smoke(*valid_files(tmp_path))


@pytest.mark.parametrize("field", ["tool_calls", "completed_tools"])
def test_agentcore_smoke_requires_completed_inspection(tmp_path: Path, field: str) -> None:
    result, state, manifest = valid_files(tmp_path)
    value = json.loads(result.read_text())
    value[field] = []
    result.write_text(json.dumps(value))
    with pytest.raises(AssertionError, match="get_invoice"):
        assert_smoke(result, state, manifest)


def test_agentcore_smoke_rejects_mutated_state(tmp_path: Path) -> None:
    result, state, manifest = valid_files(tmp_path)
    value = json.loads(state.read_text())
    value["refunds"] = 1
    state.write_text(json.dumps(value))
    with pytest.raises(AssertionError, match="changed billing mutation"):
        assert_smoke(result, state, manifest)
