"""TRL Harbor harness exposing the billing tools for Phase 8B."""

from __future__ import annotations

import base64
import json
import os
import shlex
import tempfile
import uuid
from pathlib import Path

try:  # Supports both Phase 8B's direct script runner and Phase 8C package imports.
    from .harbor_compat import trial_log_mounts
except ImportError:  # pragma: no cover - exercised by the direct remote runner
    from harbor_compat import trial_log_mounts
from trl.experimental.harbor import HarborEnv

BRIDGE_DESTINATION = "/opt/agentops/sandbox_billing_bridge.py"
BRIDGE_STAGING_PATH = "/tmp/phase8b-sandbox-billing-bridge.py"


class BillingHarborEnv(HarborEnv):
    """Harbor harness whose public methods are the complete policy tool surface."""

    PROMPT_SUFFIX = (
        "\n\nUse the billing tools to inspect the invoice and perform only the action "
        "warranted by its current state. Finish with a short response after the "
        "required tool actions."
    )

    async def _start(self, task_dir: str) -> str:
        """Start Harbor with the Trial log mounts TRL 1.13 omits."""

        from harbor.environments.factory import EnvironmentFactory
        from harbor.models.task.task import Task
        from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
        from harbor.models.trial.paths import TrialPaths

        await self._stop()
        self._task = Task(task_dir=Path(task_dir))
        trial_root = os.environ.get("PHASE8B_TRIAL_ROOT")
        if trial_root:
            Path(trial_root).mkdir(parents=True, exist_ok=True)
        self._paths = TrialPaths(
            trial_dir=Path(tempfile.mkdtemp(prefix="harbor_trl_", dir=trial_root))
        )
        self._paths.mkdir()
        self._paths.chmod_dir()
        self._env = EnvironmentFactory.create_environment_from_config(
            config=TrialEnvironmentConfig(type=self._environment_type),
            environment_dir=self._task.paths.environment_dir,
            environment_name=self._task.short_name,
            session_id=uuid.uuid4().hex,
            trial_paths=self._paths,
            task_env_config=self._task.config.environment,
            mounts=trial_log_mounts(self._paths),
        )
        await self._env.start(force_build=False)
        await self._upload_build_files()
        await self._env.run_healthcheck()
        await self._env.exec("mkdir -p /workdir /home/user/input")
        await self._setup()
        return self._task.instruction

    async def _setup(self) -> None:
        """Install the locally packaged billing runtime and bridge in the sandbox."""

        wheel = Path(os.environ.get("PHASE8B_WHEEL_PATH", ""))
        if not wheel.is_file():
            raise RuntimeError("PHASE8B_WHEEL_PATH must identify the packaged project wheel")
        wheel_destination = f"/tmp/{wheel.name}"
        bridge = Path(__file__).with_name("sandbox_billing_bridge.py")
        if not bridge.is_file():
            raise RuntimeError("sandbox billing bridge is unavailable")
        await self._env.upload_file(wheel, wheel_destination)
        await self._env.upload_file(bridge, BRIDGE_STAGING_PATH)
        directory = await self._env.exec(
            f"mkdir -p /opt/agentops && mv {BRIDGE_STAGING_PATH} {BRIDGE_DESTINATION}",
            user="root",
        )
        if directory.return_code != 0:
            output = directory.stderr or directory.stdout or "no output"
            raise RuntimeError(f"could not create sandbox bridge directory: {output}")
        result = await self._env.exec(
            f"python -m pip install --no-deps {shlex.quote(wheel_destination)}",
            timeout_sec=120,
            user="root",
        )
        if result.return_code != 0:
            output = result.stderr or result.stdout or "no output"
            raise RuntimeError(f"failed to install sandbox billing wheel: {output}")

    def get_invoice(self, invoice_id: str) -> str:
        """Inspect the current state of one invoice.

        Args:
            invoice_id: The invoice identifier to inspect.

        Returns:
            A JSON observation containing the invoice or a structured domain error.
        """

        return self._invoke("get_invoice", {"invoice_id": invoice_id})

    def refund_invoice(self, invoice_id: str, reason: str) -> str:
        """Create a refund for one invoice when that action is appropriate.

        Args:
            invoice_id: The invoice identifier to refund.
            reason: A short reason recorded with the refund.

        Returns:
            A JSON observation containing the refund result or a structured domain error.
        """

        return self._invoke("refund_invoice", {"invoice_id": invoice_id, "reason": reason})

    def escalate_dispute(self, invoice_id: str, reason: str) -> str:
        """Escalate a disputed invoice when that action is appropriate.

        Args:
            invoice_id: The invoice identifier to escalate.
            reason: A short reason recorded with the escalation.

        Returns:
            A JSON observation containing the escalation result or a structured domain error.
        """

        return self._invoke("escalate_dispute", {"invoice_id": invoice_id, "reason": reason})

    def _invoke(self, operation: str, arguments: dict[str, str]) -> str:
        payload = base64.b64encode(json.dumps(arguments, sort_keys=True).encode()).decode()
        command = shlex.join(["python", BRIDGE_DESTINATION, operation, payload])
        return self._exec(command)
