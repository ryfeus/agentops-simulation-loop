"""One deterministic, reusable Harbor agent for Phase 10 verifier calibration."""

from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from agentops_demo.benchmark.phase10_variants import NEGATIVE_STRATEGIES


class Phase10NegativeAgent(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "phase10-negative"

    def __init__(
        self,
        logs_dir: Path,
        *,
        strategy: str,
        target_invoice: str,
        wrong_invoice: str = "",
        inspection_ids: str | list[str] = "[]",
        **kwargs: object,
    ) -> None:
        super().__init__(logs_dir, **kwargs)
        if strategy not in NEGATIVE_STRATEGIES:
            raise ValueError(f"unknown Phase 10 negative strategy: {strategy}")
        parsed = json.loads(inspection_ids) if isinstance(inspection_ids, str) else inspection_ids
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("inspection_ids must be a JSON list of invoice IDs")
        self.payload = {
            "strategy": strategy,
            "target_invoice": target_invoice,
            "wrong_invoice": wrong_invoice,
            "inspection_ids": parsed,
        }

    async def install(self, environment: BaseEnvironment) -> None:
        worker = Path(__file__).with_name("phase10_negative_worker.py")
        await environment.upload_file(worker, "/tmp/phase10-negative-worker.py")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        encoded = base64.b64encode(json.dumps(self.payload, sort_keys=True).encode()).decode()
        result = await self.exec_as_agent(
            environment,
            command=shlex.join(["python", "/tmp/phase10-negative-worker.py", encoded]),
        )
        if result.return_code != 0:
            raise RuntimeError(f"Phase 10 negative worker failed: {result.stderr or result.stdout}")
        context.metadata = {"phase10_negative_strategy": self.payload["strategy"]}
