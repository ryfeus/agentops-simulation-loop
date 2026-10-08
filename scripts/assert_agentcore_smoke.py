"""Assert the direct AgentCore smoke and post-invocation DSQL state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.agentcore_package import MANIFEST_PATH, load_manifest

DEFAULT_RESULT = Path(".agentcore/smoke/result.json")
DEFAULT_STATE = Path(".agentcore/smoke/dsql-state.json")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def assert_smoke(result_path: Path, state_path: Path, manifest_path: Path) -> None:
    result = _read_object(result_path)
    state = _read_object(state_path)
    manifest = load_manifest(manifest_path)
    requested = [item.get("name") for item in result.get("tool_calls", [])]
    completed = result.get("completed_tools", [])
    if "get_invoice" not in requested or "get_invoice" not in completed:
        raise AssertionError("AgentCore smoke did not request and complete get_invoice")
    if not str(result.get("final_response", "")).strip():
        raise AssertionError("AgentCore smoke final response is blank")
    if result.get("agent_config_fingerprint") != manifest["agent_config"]["fingerprint"]:
        raise AssertionError("AgentCore smoke fingerprint does not match manifest")
    if result.get("source_revision") != manifest["source_revision"]:
        raise AssertionError("AgentCore smoke source revision does not match manifest")
    invoice = state.get("invoice")
    if not isinstance(invoice, dict) or invoice.get("status") != "disputed":
        raise AssertionError("inv-123 is missing or no longer disputed")
    if state.get("refunds") != 0 or state.get("escalations") != 0:
        raise AssertionError("non-mutating smoke changed billing mutation state")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args(argv)
    assert_smoke(args.result, args.state, args.manifest)
    print("AgentCore smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
