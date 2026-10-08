"""Paired frozen baseline versus one persisted Phase 9 LoRA adapter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from rl.phase8c.baseline_reward import baseline_reward
from rl.phase8c.evaluate_baseline import (
    ATTEMPTS_PER_PASS,
    EXPECTED_TOOLS,
    MODEL_ID,
    _declared_tools,
    _finite_metrics,
    _load_captured_rollouts,
    build_evaluation_config,
)
from rl.phase8c.execution_suite import load_execution_suite
from rl.phase9.task_set import load_task_set

from .adapter import AdapterIdentity, validate_persisted_adapter
from .compare import _summary, build_comparison

PASSES = 4


class Phase9bFailure(RuntimeError):
    def __init__(self, phase: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.phase, self.cause = phase, cause


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def _assert_frozen(trainer: Any, *, adapter: AdapterIdentity | None = None) -> None:
    if (
        int(getattr(trainer.state, "global_step", -1)) != 0
        or getattr(trainer, "optimizer", None) is not None
    ):
        raise RuntimeError("Phase 9b evaluation created training state")
    if adapter is not None:
        from .adapter import _sha

        if _sha(adapter.adapter_dir / "adapter_config.json") != adapter.config_sha256:
            raise RuntimeError("Phase 9b evaluation mutated its source adapter")
        weights = next(
            path
            for path in adapter.files
            if path in {"adapter_model.safetensors", "adapter_model.bin"}
        )
        if _sha(adapter.adapter_dir / weights) != adapter.weights_sha256:
            raise RuntimeError("Phase 9b evaluation mutated its source adapter weights")


def _policy_model(policy: str, adapter: AdapterIdentity | None) -> tuple[Any, dict[str, Any]]:
    if policy == "baseline":
        return MODEL_ID, {"policy": policy, "fresh_zero_init_lora": True}
    if adapter is None:
        raise ValueError("trained policy requires a verified source adapter")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="bfloat16")
    model = PeftModel.from_pretrained(base, adapter.adapter_dir, is_trainable=False)
    active = getattr(model, "active_adapter", None)
    return model, {
        "policy": policy,
        "adapter": adapter.value(),
        "active_adapter": active() if callable(active) else active,
        "peft_model_loaded": True,
        # TRL 1.13's colocated vLLM generation synchronizes a PeftModel by
        # merge_adapter()/streaming weights/unmerge_adapter(). This explicit
        # flag is provenance for that supported native code path.
        "vllm_synchronization": "trl_colocated_peft_merge_stream_unmerge",
    }


def _pass_command(
    *,
    policy: str,
    evaluation_pass: int,
    dataset_root: Path,
    output: Path,
    run_id: str,
    seed: int,
    source_run: str,
    source_dir: Path,
    task_set_path: Path,
) -> list[str]:
    if policy not in {"baseline", "trained"} or evaluation_pass not in range(PASSES):
        raise ValueError("invalid Phase 9b policy or pass")
    return [
        sys.executable,
        "-m",
        "rl.phase9b.evaluate",
        "--dataset-root",
        str(dataset_root),
        "--output",
        str(output),
        "--run-id",
        run_id,
        "--source-run",
        source_run,
        "--source-dir",
        str(source_dir),
        "--task-set",
        str(task_set_path),
        "--seed",
        str(seed),
        "--worker-policy",
        policy,
        "--worker-pass",
        str(evaluation_pass),
    ]


def _evaluate_policy(
    *,
    policy: str,
    dataset: Any,
    dataset_root: Path,
    output: Path,
    run_id: str,
    seed: int,
    adapter: AdapterIdentity | None,
    source_run: str,
    source_dir: Path,
    task_set_path: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    all_rollouts: list[dict[str, Any]] = []
    last_metrics: dict[str, Any] = {}
    proof = (
        {"policy": policy, "fresh_zero_init_lora": True}
        if policy == "baseline"
        else {"policy": policy, **(adapter.value() if adapter is not None else {})}
    )
    _write(
        output / "policy-config.json",
        {
            "schema_version": "1",
            **proof,
            "passes": PASSES,
            "attempts_per_pass": ATTEMPTS_PER_PASS,
            "seed": seed,
        },
    )
    for evaluation_pass in range(PASSES):
        # A colocated vLLM engine retains CUDA memory after GRPOTrainer is
        # collected. Each native evaluate() pass must exit its process before
        # the next engine starts on the same 24 GiB L4.
        subprocess.run(
            _pass_command(
                policy=policy,
                evaluation_pass=evaluation_pass,
                dataset_root=dataset_root,
                output=output,
                run_id=run_id,
                seed=seed,
                source_run=source_run,
                source_dir=source_dir,
                task_set_path=task_set_path,
            ),
            check=True,
        )
        rows = [
            json.loads(line)
            for line in (output / f"rollouts.pass-{evaluation_pass}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        all_rollouts.extend(rows)
        last_metrics = json.loads(
            (output / f"pass-{evaluation_pass}-result.json").read_text(encoding="utf-8")
        )["eval_metrics"]
        proof = json.loads(
            (output / f"pass-{evaluation_pass}-result.json").read_text(encoding="utf-8")
        )["model_proof"]
    if list(output.glob("checkpoint-*")):
        raise RuntimeError("Phase 9b evaluation emitted a checkpoint")
    _write_jsonl(output / "rollouts.jsonl", all_rollouts)
    task_summary = _summary(all_rollouts, {str(row["task_id"]) for row in all_rollouts})
    _write(output / "task-summary.json", task_summary)
    value = {
        "schema_version": "1",
        "status": "PASS",
        "policy": policy,
        "training_performed": False,
        "global_step": 0,
        "optimizer_created": False,
        "checkpoint_emitted": False,
        "expected_rollouts": len(dataset) * PASSES * ATTEMPTS_PER_PASS,
        "valid_rollouts": len(all_rollouts),
        "passes": PASSES,
        "attempts_per_pass": ATTEMPTS_PER_PASS,
        "attempts_per_task": PASSES * ATTEMPTS_PER_PASS,
        "eval_metrics": _finite_metrics(last_metrics),
        "task_summary": "task-summary.json",
        **proof,
    }
    if value["valid_rollouts"] != value["expected_rollouts"]:
        raise RuntimeError("Phase 9b policy did not produce complete four-pass coverage")
    expected = set(range(PASSES * ATTEMPTS_PER_PASS))
    by_task: dict[str, set[int]] = {}
    for row in all_rollouts:
        task_id, attempt = str(row["task_id"]), int(row["attempt"])
        by_task.setdefault(task_id, set())
        if attempt in by_task[task_id]:
            raise RuntimeError("Phase 9b policy produced a duplicate logical attempt")
        by_task[task_id].add(attempt)
    if set(by_task) != {str(row["task_id"]) for row in dataset} or any(
        attempts != expected for attempts in by_task.values()
    ):
        raise RuntimeError("Phase 9b policy did not produce exactly 16 attempts per task")
    _write(output / "evaluation-result.json", value)
    return value


def _evaluate_pass_worker(
    *,
    policy: str,
    evaluation_pass: int,
    dataset_root: Path,
    output: Path,
    run_id: str,
    source_run: str,
    source_dir: Path,
    task_set_path: Path,
    seed: int,
) -> None:
    """Run one native TRL pass; process exit releases its colocated vLLM engine."""

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOTrainer
    from trl.experimental.harbor import HarborSpec

    from rl.phase8c.baseline_env import BaselineBillingHarborEnv

    if policy not in {"baseline", "trained"} or evaluation_pass not in range(PASSES):
        raise ValueError("invalid Phase 9b worker policy or pass")
    suite = load_execution_suite(dataset_root / "execution-suite.json")
    task_set = load_task_set(task_set_path, suite)
    if len(task_set.training_ids) != 4 or len(task_set.regression_ids) != 1:
        raise RuntimeError("Phase 9b worker requires four training tasks and one control")
    adapter = validate_persisted_adapter(
        source_dir,
        source_run=source_run,
        execution_suite_sha256=str(suite["execution_suite_sha256"]),
    )
    spec = HarborSpec(str(dataset_root), agent=BaselineBillingHarborEnv, environment_type="docker")
    selected = task_set.evaluation_ids
    data = [
        {
            **row,
            "task_id": Path(str(row["task_dir"])).name,
            **suite["tasks"][Path(str(row["task_dir"])).name],
        }
        for row in spec.train_dataset
        if Path(str(row["task_dir"])).name in selected
    ]
    if {row["task_id"] for row in data} != selected:
        raise RuntimeError("Phase 9b worker dataset differs from selected tasks")
    dataset = Dataset.from_list(data)
    model, proof = _policy_model(policy, adapter if policy == "trained" else None)
    capture = output / f"rollouts.pass-{evaluation_pass}.raw.jsonl"
    os.environ["PHASE8C_ROLLOUT_CAPTURE"] = str(capture)
    config = build_evaluation_config(
        output=output, attempts_per_pass=ATTEMPTS_PER_PASS, seed=seed + evaluation_pass
    )
    kwargs: dict[str, Any] = {}
    if policy == "baseline":
        kwargs["peft_config"] = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        eval_dataset=dataset,
        environment_factory=spec.environment_factory,
        reward_funcs=baseline_reward,
        **kwargs,
    )
    _assert_frozen(trainer, adapter=adapter if policy == "trained" else None)
    metrics = trainer.evaluate()
    _assert_frozen(trainer, adapter=adapter if policy == "trained" else None)
    if list(output.glob("checkpoint-*")):
        raise RuntimeError("Phase 9b worker emitted a checkpoint")
    rows = _load_captured_rollouts(
        capture,
        run_id=run_id,
        evaluation_pass=evaluation_pass,
        attempts_per_pass=ATTEMPTS_PER_PASS,
    )
    for row in rows:
        row["policy"] = policy
        row["rollout_id"] = f"{policy}:{row['task_id']}:{int(row['attempt']):02d}"
    _write_jsonl(output / f"rollouts.pass-{evaluation_pass}.jsonl", rows)
    _write(
        output / f"pass-{evaluation_pass}-result.json",
        {
            "schema_version": "1",
            "policy": policy,
            "evaluation_pass": evaluation_pass,
            "seed": seed + evaluation_pass,
            "valid_rollouts": len(rows),
            "eval_metrics": _finite_metrics(metrics),
            "model_proof": proof,
            "training_performed": False,
            "global_step": 0,
            "optimizer_created": False,
            "checkpoint_emitted": False,
        },
    )


def run(
    *,
    dataset_root: Path,
    output: Path,
    run_id: str,
    source_run: str,
    source_dir: Path,
    task_set_path: Path,
    seed: int = 20260923,
) -> dict[str, Any]:
    """Evaluate baseline fully before releasing it and evaluating the persisted adapter."""

    if _declared_tools() != EXPECTED_TOOLS:
        raise ValueError("Phase 9b model-visible tool surface differs from Phase 8B")
    phase, started = "TASK_SET_INVALID", time.perf_counter()
    try:
        from datasets import Dataset
        from trl.experimental.harbor import HarborSpec

        from rl.phase8c.baseline_env import BaselineBillingHarborEnv

        suite = load_execution_suite(dataset_root / "execution-suite.json")
        task_set = load_task_set(task_set_path, suite)
        if len(task_set.training_ids) != 4 or len(task_set.regression_ids) != 1:
            raise RuntimeError("Phase 9b requires exactly four training tasks and one control")
        adapter = validate_persisted_adapter(
            source_dir,
            source_run=source_run,
            execution_suite_sha256=str(suite["execution_suite_sha256"]),
        )
        _write(output / "source-adapter.json", adapter.value())
        phase = "HARBOR_TASK_INVALID"
        spec = HarborSpec(
            str(dataset_root), agent=BaselineBillingHarborEnv, environment_type="docker"
        )
        rows = [spec.train_dataset[index] for index in range(len(spec.train_dataset))]
        selected = task_set.evaluation_ids
        data = [
            {
                **row,
                "task_id": Path(str(row["task_dir"])).name,
                **suite["tasks"][Path(str(row["task_dir"])).name],
            }
            for row in rows
            if Path(str(row["task_dir"])).name in selected
        ]
        if {row["task_id"] for row in data} != selected:
            raise RuntimeError("Phase 9b dataset is not exactly the selected five tasks")
        dataset = Dataset.from_list(data)
        _write(
            output / "evaluation-config.json",
            {
                "schema_version": "1",
                "model_id": MODEL_ID,
                "source_run": source_run,
                "execution_suite_sha256": suite["execution_suite_sha256"],
                **adapter.value(),
                "training_task_ids": sorted(task_set.training_ids),
                "control_task_id": next(iter(task_set.regression_ids)),
                "passes": PASSES,
                "attempts_per_pass": ATTEMPTS_PER_PASS,
                "seed": seed,
                "seed_schedule": [seed + offset for offset in range(PASSES)],
                "attempts_per_task_per_policy": PASSES * ATTEMPTS_PER_PASS,
                "policies": ["baseline", "trained"],
            },
        )
        phase = "BASELINE_EVALUATION"
        baseline = _evaluate_policy(
            policy="baseline",
            dataset=dataset,
            dataset_root=dataset_root,
            output=output / "baseline",
            run_id=run_id,
            seed=seed,
            adapter=None,
            source_run=source_run,
            source_dir=source_dir,
            task_set_path=task_set_path,
        )
        phase = "TRAINED_EVALUATION"
        trained = _evaluate_policy(
            policy="trained",
            dataset=dataset,
            dataset_root=dataset_root,
            output=output / "trained",
            run_id=run_id,
            seed=seed,
            adapter=adapter,
            source_run=source_run,
            source_dir=source_dir,
            task_set_path=task_set_path,
        )
        phase = "REPORTING"

        def read(path: Path) -> list[dict[str, Any]]:
            return [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            ]

        comparison = build_comparison(
            baseline=read(output / "baseline" / "rollouts.jsonl"),
            trained=read(output / "trained" / "rollouts.jsonl"),
            training_ids=task_set.training_ids,
            control_id=next(iter(task_set.regression_ids)),
            destination=output,
        )
    except Exception as exc:
        raise Phase9bFailure(phase, exc) from exc
    result = {
        "schema_version": "1",
        "phase": "9b",
        "status": "PASS",
        "source_run": source_run,
        "execution_suite_sha256": suite["execution_suite_sha256"],
        "baseline": baseline,
        "trained": trained,
        "comparison": comparison,
        "human_review_required": True,
        "wall_time_seconds": time.perf_counter() - started,
    }
    _write(output / "evaluation-result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--task-set", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--worker-policy", choices=("baseline", "trained"))
    parser.add_argument("--worker-pass", type=int)
    args = parser.parse_args(argv)
    if args.worker_policy is not None or args.worker_pass is not None:
        if args.worker_policy is None or args.worker_pass is None:
            parser.error("--worker-policy and --worker-pass must be supplied together")
        _evaluate_pass_worker(
            policy=args.worker_policy,
            evaluation_pass=args.worker_pass,
            dataset_root=args.dataset_root,
            output=args.output,
            run_id=args.run_id,
            source_run=args.source_run,
            source_dir=args.source_dir,
            task_set_path=args.task_set,
            seed=args.seed,
        )
        return 0
    try:
        print(
            json.dumps(
                run(
                    dataset_root=args.dataset_root,
                    output=args.output,
                    run_id=args.run_id,
                    source_run=args.source_run,
                    source_dir=args.source_dir,
                    task_set_path=args.task_set,
                    seed=args.seed,
                ),
                sort_keys=True,
            )
        )
    except Phase9bFailure as exc:
        _write(
            args.output / "evaluation-result.json",
            {
                "schema_version": "1",
                "phase": "9b",
                "status": "FAIL",
                "failure_phase": exc.phase,
                "error_type": type(exc.cause).__name__,
                "error": str(exc.cause),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
