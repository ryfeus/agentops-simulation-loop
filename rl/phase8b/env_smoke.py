"""Deterministic Harbor verifier preflight for the single Phase 8B task."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from billing_harbor_env import BillingHarborEnv

DEBUG_COMMANDS = (
    "ls -la /logs/verifier",
    "cat /logs/verifier/reward.json",
    "cat /logs/verifier/reward.txt",
    "cat /logs/verifier/test-stdout.txt",
    "ls -la /tests",
    "ls -la /app/data",
    "test -f /app/data/billing.db",
    "test -f /app/data/agent-run.json",
)


class PreflightFailure(RuntimeError):
    """Preserve the stable preflight phase without hiding the Harbor cause."""

    def __init__(self, phase: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.phase = phase
        self.cause = cause


def _close(environment: BillingHarborEnv) -> None:
    """Stop the private Harbor lifecycle without exposing cleanup as a model tool."""

    environment._run(environment._stop())


def _bounded(value: str | None, limit: int = 8000) -> str:
    text = value or ""
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


def _capture_verifier_debug(environment: BillingHarborEnv, destination: Path) -> None:
    """Capture best-effort container and TrialPaths evidence before teardown."""

    paths = environment._paths
    value: dict[str, Any] = {"schema_version": "1", "commands": []}
    if paths is not None:
        value["host_paths"] = {
            "trial_dir": str(paths.trial_dir),
            "verifier_dir": str(paths.verifier_dir),
            "reward_json_path": str(paths.reward_json_path),
            "reward_text_path": str(paths.reward_text_path),
            "trial_dir_exists": paths.trial_dir.exists(),
            "verifier_dir_exists": paths.verifier_dir.exists(),
            "reward_json_exists": paths.reward_json_path.exists(),
            "reward_text_exists": paths.reward_text_path.exists(),
        }
    for command in DEBUG_COMMANDS:
        record: dict[str, Any] = {"command": command}
        try:
            result = environment._run(environment._env.exec(command, timeout_sec=20))
            record.update(
                {
                    "return_code": result.return_code,
                    "stdout": _bounded(result.stdout),
                    "stderr": _bounded(result.stderr),
                }
            )
        except Exception as exc:
            record.update({"error_type": type(exc).__name__, "error": str(exc)})
        value["commands"].append(record)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _reset(environment: BillingHarborEnv, task: Path) -> None:
    try:
        environment.reset(str(task))
    except Exception as exc:
        raise PreflightFailure("HARBOR_SANDBOX_START", exc) from exc


def _reward(environment: BillingHarborEnv, debug_path: Path) -> float:
    try:
        reward = float(environment.reward)
        paths = environment._paths
        if paths is None or not paths.reward_json_path.is_file():
            raise RuntimeError("Harbor verifier did not persist reward.json to TrialPaths")
        return reward
    except Exception as exc:
        _capture_verifier_debug(environment, debug_path)
        raise PreflightFailure("HARBOR_VERIFIER", exc) from exc


def _run_noop(task: Path, debug_path: Path) -> tuple[float, bool]:
    environment = BillingHarborEnv(environment_type="docker")
    try:
        _reset(environment, task)
        reward = _reward(environment, debug_path)
        return reward, bool(environment._paths and environment._paths.reward_json_path.is_file())
    finally:
        _close(environment)


def _run_correct(task: Path, debug_path: Path) -> tuple[float, dict[str, Any], bool]:
    environment = BillingHarborEnv(environment_type="docker")
    try:
        _reset(environment, task)
        try:
            invoice = json.loads(environment.get_invoice("inv-201"))
            if not invoice.get("ok") or invoice.get("invoice", {}).get("status") != "paid":
                raise RuntimeError("paid-refund-direct did not begin with paid inv-201")
            refund = json.loads(
                environment.refund_invoice("inv-201", "Phase 8B deterministic preflight")
            )
            if not refund.get("ok"):
                raise RuntimeError(f"deterministic refund failed: {refund}")
        except Exception as exc:
            raise PreflightFailure("BILLING_BRIDGE", exc) from exc
        reward = _reward(environment, debug_path)
        persisted = bool(environment._paths and environment._paths.reward_json_path.is_file())
        return reward, invoice, persisted
    finally:
        _close(environment)


def _write_summary(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    debug_path = args.summary.with_name("harbor-verifier-debug.json")
    try:
        noop, noop_reward_json_exists = _run_noop(args.task, debug_path)
        correct, invoice, correct_reward_json_exists = _run_correct(args.task, debug_path)
        if noop != 0.0 or correct != 1.0:
            raise PreflightFailure(
                "HARBOR_VERIFIER",
                RuntimeError(f"unexpected Harbor rewards: noop={noop}, correct={correct}"),
            )
        value: dict[str, Any] = {
            "schema_version": "1",
            "status": "PASS",
            "task_id": args.task.name,
            "noop_reward": noop,
            "correct_reward": correct,
            "noop_reward_json_exists": noop_reward_json_exists,
            "correct_reward_json_exists": correct_reward_json_exists,
            "initial_invoice": invoice["invoice"],
        }
    except PreflightFailure as exc:
        value = {
            "schema_version": "1",
            "status": "FAIL",
            "failure_phase": exc.phase,
            "error_type": type(exc.cause).__name__,
            "error": str(exc.cause),
            "traceback": traceback.format_exc(),
        }
        _write_summary(args.summary, value)
        raise
    except Exception as exc:
        value = {
            "schema_version": "1",
            "status": "FAIL",
            "failure_phase": "HARBOR_SANDBOX_START",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_summary(args.summary, value)
        raise
    _write_summary(args.summary, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
