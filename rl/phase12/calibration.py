"""Executable deterministic calibration, separate from model evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from rl.phase12.common import write
from rl.phase12.corpus import check, render, scenarios
from rl.phase12.demonstrations import conversation


def calibrate(output: Path) -> dict[str, Any]:
    """Run correct tool sequences and no-op negatives in real Harbor sandboxes.

    This is an explicit Docker-backed execution command, not part of config-check.
    Only pass/fail and lineage leave this isolated task; no test conversations are
    supplied to model-training payloads.
    """
    from rl.phase12.environment import ComparisonEnv

    if output.exists():
        raise FileExistsError(output)
    frozen = check()
    data = scenarios()
    results = {}
    with tempfile.TemporaryDirectory(prefix="phase12-calibrate-") as temporary:
        root = Path(temporary) / "datasets"
        render(root, ("train", "dev", "test"))
        env = ComparisonEnv(environment_type="docker")
        try:
            for role, tasks in data.items():
                for key, scenario in tasks.items():
                    task = root / role / "execution/tasks" / key
                    good = conversation(env, scenario, task)["reward"]
                    env.reset(task_dir=str(task))
                    noop = float(env.reward)
                    # A plausible wrong write must fail even when it happens to leave
                    # an idempotent final state: trajectory rules are part of reward.
                    env.reset(task_dir=str(task))
                    protected = next(
                        (
                            i.invoice_id
                            for i in scenario.expected_invariants
                            if i.type == "unchanged"
                        ),
                        scenario.initial_state.invoices[0].id,
                    )
                    env.refund_invoice(protected, "Calibration wrong write")
                    wrong_write = float(env.reward)
                    results[key] = {"correct": good, "noop": noop, "wrong_write": wrong_write}
                    if (good, noop, wrong_write) != (1.0, 0.0, 0.0):
                        raise ValueError(f"calibration failed: {key}")
        finally:
            env._run(env._stop())
    result = {
        "execution_valid": True,
        "corpus_sha256": frozen["corpus_sha256"],
        "backend": "harbor-docker-billing-tools",
        "tasks": results,
    }
    write(output, result)
    return result


def validate(value: dict[str, Any], corpus: dict[str, Any]) -> None:
    expected = {key for role in corpus["roles"].values() for key in role}
    if (
        value.get("execution_valid") is not True
        or value.get("corpus_sha256") != corpus["corpus_sha256"]
        or value.get("backend") != "harbor-docker-billing-tools"
        or set(value.get("tasks", {})) != expected
        or any(
            v != {"correct": 1.0, "noop": 0.0, "wrong_write": 0.0} for v in value["tasks"].values()
        )
    ):
        raise ValueError("complete executable calibration is required before training")
