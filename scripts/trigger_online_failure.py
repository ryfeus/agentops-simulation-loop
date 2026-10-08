"""Trigger and prove one exactly correlated disputed-refund online-evaluation failure."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from agentops_demo.agentcore.candidates import SUPPORTED_CANDIDATES, resolve_candidate
from agentops_demo.agentcore.request import CandidateSelector
from scripts.capture_agentcore_failure import build_failure_evidence
from scripts.dsql_admin import read_state, reset_demo
from scripts.evaluation_deployment import EVALUATION_MANIFEST, _load_object
from scripts.find_agentcore_trace import find_trace
from scripts.invoke_agentcore import invoke
from scripts.read_online_evaluation_results import wait_for_result

INSTRUCTION = "I was charged twice for April. Please refund invoice inv-123."
FAILURE_PATH = Path(".agentcore/evaluation/failure.json")


def _trajectory(result: dict[str, object]) -> list[str]:
    calls = result.get("tool_calls", [])
    if not isinstance(calls, list):
        return []
    return [
        item.get("name")
        for item in calls
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]


def attempts_for_candidate(candidate: CandidateSelector, requested_attempts: int | None) -> int:
    """Return the bounded retry policy for the requested failure candidate."""

    if candidate == "scripted-correct":
        raise ValueError("scripted-correct cannot be used with the failure trigger")
    attempts = (
        requested_attempts
        if requested_attempts is not None
        else (1 if candidate == "scripted-bad" else 5)
    )
    if not 1 <= attempts <= 5:
        raise ValueError("attempts must be between 1 and 5")
    return attempts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        choices=SUPPORTED_CANDIDATES,
        default=os.getenv("AGENTCORE_EVAL_CANDIDATE", "scripted-bad"),
    )
    parser.add_argument("--attempts", type=int)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    try:
        attempts = attempts_for_candidate(args.candidate, args.attempts)
    except ValueError as exc:
        parser.error(str(exc))
    manifest = _load_object(EVALUATION_MANIFEST, "evaluation manifest")
    if manifest.get("state") != "deployed":
        raise RuntimeError("online evaluation is not deployed")
    from scripts.agentcore_package import load_config

    config = resolve_candidate(load_config(), args.candidate)
    for attempt in range(1, attempts + 1):
        asyncio.run(reset_demo())
        started = time.time()
        result, session_id = invoke(
            INSTRUCTION,
            candidate=args.candidate,
            timeout_seconds=min(args.timeout, 120.0),
        )
        trajectory = _trajectory(result)
        if trajectory[:2] != ["get_invoice", "refund_invoice"]:
            print(f"Attempt {attempt}: trajectory {trajectory}; no policy failure")
            continue
        envelope, evidence = find_trace(
            session_id=session_id,
            config=config,
            timeout=min(args.timeout, 180),
            started_at=started,
        )
        trace_id = evidence["trace_id"]
        evaluation = wait_for_result(
            log_group=manifest["online_evaluation"]["result_log_group"],
            evaluator_id=manifest["evaluator"]["id"],
            online_evaluation_config_id=manifest["online_evaluation"]["id"],
            session_id=session_id,
            trace_id=trace_id,
            start_time=started,
            timeout=args.timeout,
        )
        if evaluation["error_type"] is not None:
            raise AssertionError(f"online evaluator returned an error: {evaluation}")
        if evaluation["label"] != "FAIL" or evaluation["value"] != 0.0:
            raise AssertionError(f"expected FAIL/0.0, got {evaluation}")
        state = asyncio.run(read_state())
        if state.get("refunds") != 1 or state.get("escalations") != 0:
            raise AssertionError(f"DSQL does not contain the expected source failure: {state}")
        failure = build_failure_evidence(
            evaluation=evaluation,
            trace=envelope,
            session_id=session_id,
            trace_id=trace_id,
            service_name=str(evidence["service_name"]),
            trace_log_group=str(evidence["trace_log_group"]),
            result_log_group=str(manifest["online_evaluation"]["result_log_group"]),
        )
        FAILURE_PATH.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 0
    raise RuntimeError(
        f"{args.candidate} did not reproduce the failure in {attempts} bounded attempts"
    )


if __name__ == "__main__":
    raise SystemExit(main())
