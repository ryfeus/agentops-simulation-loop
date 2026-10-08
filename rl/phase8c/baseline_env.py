"""Private evidence/timing instrumentation for frozen Harbor evaluation."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from rl.phase8b.billing_harbor_env import BillingHarborEnv


class BaselineBillingHarborEnv(BillingHarborEnv):
    """Phase 8B harness with no additional model-visible methods or tools."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._baseline_reset_ms = 0.0
        self._baseline_tool_ms = 0.0
        self._baseline_tool_count = 0
        self._baseline_verifier_ms = 0.0
        self._baseline_retry_count = 0
        self._baseline_prior_infrastructure_failures: list[str] = []

    @staticmethod
    def _baseline_retryable_start_failure(error: Exception) -> str | None:
        """Classify only lifecycle errors; policy decisions are never retried here."""

        text = str(error).lower()
        if "docker daemon" in text or "cannot connect to docker" in text:
            return "DOCKER_UNAVAILABLE"
        if "harbor" in text or "sandbox" in text or "timed out" in text:
            return "HARBOR_SANDBOX_START"
        return None

    def reset(self, *args: Any, **kwargs: Any) -> str:
        retries = int(os.environ.get("PHASE8C_INFRA_RETRIES", "2"))
        self._baseline_retry_count = 0
        self._baseline_prior_infrastructure_failures = []
        self._baseline_tool_ms = 0.0
        self._baseline_tool_count = 0
        self._baseline_verifier_ms = 0.0
        started = time.perf_counter()
        for retry in range(retries + 1):
            try:
                return super().reset(*args, **kwargs)
            except Exception as exc:
                failure = self._baseline_retryable_start_failure(exc)
                if failure is None:
                    raise
                if retry >= retries:
                    raise
                self._baseline_retry_count += 1
                self._baseline_prior_infrastructure_failures.append(failure)
            finally:
                self._baseline_reset_ms = (time.perf_counter() - started) * 1000
        raise RuntimeError("unreachable Phase 8C reset retry state")

    def _invoke(self, operation: str, arguments: dict[str, str]) -> str:
        started = time.perf_counter()
        try:
            return super()._invoke(operation, arguments)
        finally:
            self._baseline_tool_ms += (time.perf_counter() - started) * 1000
            self._baseline_tool_count += 1

    @property
    def reward(self) -> float:
        started = time.perf_counter()
        try:
            return super().reward
        finally:
            self._baseline_verifier_ms += (time.perf_counter() - started) * 1000

    def _baseline_evidence(self) -> dict[str, Any]:
        """Read semantic verifier artifacts before TRL recycles this environment."""

        paths = self._paths
        evidence: dict[str, Any] = {
            "environment_reset_ms": self._baseline_reset_ms,
            "tool_execution_ms": self._baseline_tool_ms,
            "tool_call_count": self._baseline_tool_count,
            "verifier_ms": self._baseline_verifier_ms,
            "retry_count": self._baseline_retry_count,
            "prior_infrastructure_failures": self._baseline_prior_infrastructure_failures,
            "verifier_components": {},
            "verifier_diagnostics": {},
            "tool_calls": [],
        }
        if paths is not None:
            for name, path in (
                ("reward_json", paths.reward_json_path),
                ("verifier_diagnostics", paths.verifier_dir / "diagnostics.json"),
            ):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if name == "reward_json" and isinstance(data, dict):
                    evidence["verifier_components"] = data
                elif name == "verifier_diagnostics":
                    evidence["verifier_diagnostics"] = data
        try:
            result = self._run(self._env.exec("cat /app/data/agent-run.json", timeout_sec=20))
            trajectory = json.loads(result.stdout or "{}")
            if isinstance(trajectory, dict):
                calls = trajectory.get("tool_calls", [])
                evidence["tool_calls"] = calls if isinstance(calls, list) else []
        except Exception:  # best-effort evidence must not change verifier semantics
            pass
        return evidence
