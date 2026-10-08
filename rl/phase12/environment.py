"""Private instrumentation; the three existing billing methods remain the only tools."""

from __future__ import annotations

from typing import Any

from rl.phase8c.baseline_env import BaselineBillingHarborEnv


class ComparisonEnv(BaselineBillingHarborEnv):
    def _exec(self, command: str, timeout: int = 180) -> str:
        # Preserve Harbor's transport response exactly, while retaining the exit
        # status that its string-only _exec interface otherwise discards.
        result = self._run(self._env.exec(command, timeout_sec=timeout))
        self._phase12_last_return_code = result.return_code
        out = (result.stdout or "") + (result.stderr or "")
        if len(out) > 8000:
            out = out[:8000] + "\n... [truncated]"
        return out or f"(empty output, rc={result.return_code})"

    async def _stop(self) -> None:
        # Tool subprocesses and the verifier have returned before teardown.
        # Harbor's idle PID-1 shell otherwise consumes Docker's ten-second
        # grace period on every rollout. Keep native cleanup, including mounts,
        # volumes and networks, and shorten only this post-rollout wait.
        env = self._env
        if env is not None and self._environment_type == "docker":
            original = env._run_docker_compose_command

            async def shutdown_timeout(command: list[str], *args: Any, **kwargs: Any) -> Any:
                if command and command[0] in {"stop", "down"} and "--timeout" not in command:
                    command = [command[0], "--timeout", "1", *command[1:]]
                return await original(command, *args, **kwargs)

            env._run_docker_compose_command = shutdown_timeout
        await super()._stop()

    def reset(self, *args: Any, **kwargs: Any) -> str:
        self._phase12_observations: list[dict[str, Any]] = []
        self._phase12_generation: dict[str, Any] = {}
        return super().reset(*args, **kwargs)

    def _invoke(self, operation: str, arguments: dict[str, str]) -> str:
        self._phase12_last_return_code = None
        result = super()._invoke(operation, arguments)
        # argparse rejects these requests before the bridge records a logical
        # billing call. They are policy tool errors, not missing trajectory data.
        rejected = self._phase12_last_return_code == 2 and any(
            result.rstrip().endswith(f"error: {message}")
            for message in (
                "tool arguments must be strings",
                "invoice_id is required",
                "reason is required",
            )
        )
        self._phase12_observations.append(
            {
                "name": operation,
                "arguments": arguments,
                "response": result,
                "return_code": self._phase12_last_return_code,
                "bridge_rejected": rejected,
            }
        )
        return result

    def _phase12_executed_calls(self) -> list[dict[str, Any]]:
        return [
            {"name": observation["name"], "arguments": observation["arguments"]}
            for observation in self._phase12_observations
            if not observation["bridge_rejected"]
        ]
