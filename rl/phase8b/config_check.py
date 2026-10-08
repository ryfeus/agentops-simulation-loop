"""Cheap locked-runtime contract check for Phase 8B Harbor wiring."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

from billing_harbor_env import BillingHarborEnv
from execution_task import EXECUTION_PROFILE, derive_execution_task
from harbor_compat import trial_log_mounts

TASK_ID = "paid-refund-direct"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASK_SOURCE = REPOSITORY_ROOT / "benchmarks" / "billing" / "tasks" / TASK_ID
EXPECTED_TOOLS = {"get_invoice", "refund_invoice", "escalate_dispute"}


def declared_tools() -> set[str]:
    """Return exactly the tool methods declared by this harness."""

    return {
        name
        for name, member in BillingHarborEnv.__dict__.items()
        if not name.startswith("_") and inspect.isfunction(member)
    }


def main() -> int:
    from harbor.models.trial.paths import TrialPaths
    from trl.experimental.harbor import HarborSpec

    if not TASK_SOURCE.is_dir():
        raise RuntimeError(f"missing Harbor task {TASK_SOURCE}")
    if declared_tools() != EXPECTED_TOOLS:
        raise RuntimeError(f"unexpected billing tool surface: {sorted(declared_tools())}")
    with tempfile.TemporaryDirectory(prefix="phase8b-config-") as temporary:
        dataset_root = Path(temporary)
        execution = derive_execution_task(TASK_SOURCE, dataset_root / "tasks" / TASK_ID)
        if execution.execution_profile != EXECUTION_PROFILE:
            raise RuntimeError("Phase 8B execution profile is not explicit")
        if execution.base_task_sha256 == execution.execution_task_sha256:
            raise RuntimeError("Phase 8B execution task must differ from canonical task")
        paths = TrialPaths(trial_dir=dataset_root / "trial")
        expected_mounts = {
            str(paths.verifier_dir.resolve()): "/logs/verifier",
            str(paths.agent_dir.resolve()): "/logs/agent",
            str(paths.artifacts_dir.resolve()): "/logs/artifacts",
        }
        observed_mounts = {mount["source"]: mount["target"] for mount in trial_log_mounts(paths)}
        if observed_mounts != expected_mounts:
            raise RuntimeError("Phase 8B Trial log mount contract is invalid")
        spec = HarborSpec(str(dataset_root), agent=BillingHarborEnv, environment_type="docker")
        if len(spec.train_dataset) != 1:
            raise RuntimeError("Phase 8B HarborSpec must expose exactly one task")
        if not callable(spec.environment_factory) or not spec.reward_funcs:
            raise RuntimeError("HarborSpec did not expose environment/reward hooks")
        row = spec.train_dataset[0]
        if Path(str(row["task_dir"])).name != TASK_ID:
            raise RuntimeError("HarborSpec resolved an unintended task")
    print("Phase 8B locked Harbor configuration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
