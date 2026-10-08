"""Per-rollout diagnostics and a fail-closed, train-only budget selection gate."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from rl.phase12.common import fingerprint


def inspect_completion(
    ids: list[int],
    messages: list[dict[str, Any]],
    raw: str,
    stop_ids: set[int],
    budget: int,
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    clipped = not ids or ids[-1] not in stop_ids
    pending = bool(messages and messages[-1].get("tool_calls"))
    malformed = 0
    # The raw decoded transcript includes tool results: only inspect assistant content
    # left unparsed by the native parser. Valid structured tool calls are counted separately.
    for message in messages:
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if "<tool_call>" in content and not message.get("tool_calls"):
                malformed += 1
    errors = 0
    for message in messages:
        if message.get("role") == "tool":
            try:
                result = json.loads(message.get("content", "{}"))
                errors += int(
                    isinstance(result, dict) and (result.get("ok") is False or "error" in result)
                )
            except (TypeError, ValueError):
                errors += 1
    return {
        "completion_token_count": len(ids),
        "token_budget": budget,
        "clipped": clipped,
        "clipped_before_first_tool": clipped and not calls,
        "termination_reason": "length"
        if clipped and len(ids) >= budget
        else "unterminated"
        if clipped
        else "pending_tool"
        if pending
        else "eos",
        "pending_tool_call": pending,
        "parser_failures": malformed,
        "tool_errors": errors,
        "raw_completion": raw,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("missing rollout evidence")
    return {
        "rollouts": len(rows),
        "clipped_fraction": sum(r["clipped"] for r in rows) / len(rows),
        "clipped_before_first_tool": sum(r["clipped_before_first_tool"] for r in rows),
        "parser_failures": sum(r["parser_failures"] for r in rows),
        "tool_errors": sum(r["tool_errors"] for r in rows),
        "completion_tokens": sum(r["completion_token_count"] for r in rows),
        "termination_reasons": dict(Counter(r["termination_reason"] for r in rows)),
    }


def select_budget(
    evidence: dict[str, dict[str, Any]],
    representatives: list[str],
    max_calls: int,
    exp: dict[str, Any],
) -> dict[str, Any]:
    if max_calls + 1 > exp["rollout"]["max_tool_calling_iterations"]:
        raise ValueError(
            "tool iteration bound cannot accommodate required calls and final response"
        )
    required = ["thinking-256", "nonthinking-256", "nonthinking-1024"]
    for name in required:
        if name not in evidence:
            raise ValueError(f"missing diagnostic arm: {name}")
    for arm in evidence.values():
        if (
            sorted(arm["task_ids"]) != sorted(representatives)
            or arm["attempts"] != exp["diagnostic_attempts"]
            or arm["diagnostics"]["rollouts"] != len(representatives) * exp["diagnostic_attempts"]
            or arm.get("execution_valid") is not True
        ):
            raise ValueError("diagnostic evidence has invalid train-only coverage")
    threshold = exp["rollout"]["max_clipped_fraction"]
    if (
        evidence["nonthinking-1024"]["diagnostics"]["clipped_fraction"] > threshold
        and "nonthinking-2048" not in evidence
    ):
        raise ValueError("2048-token diagnostic arm is required")
    for budget in exp["rollout"]["budgets"]:
        arm = evidence.get(f"nonthinking-{budget}")
        if arm and arm["diagnostics"]["clipped_fraction"] <= threshold:
            return {
                "execution_valid": True,
                "budget": budget,
                "thinking": False,
                "evidence_sha256": fingerprint(evidence),
                "experiment_sha256": fingerprint(exp),
            }
    raise ValueError("NO_QUALIFYING_ROLLOUT_CONFIGURATION: stop before training")
