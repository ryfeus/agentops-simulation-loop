"""GPU worker functions. Imported lazily by explicit execution commands only."""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rl.phase9.train_overfit import _adapter_delta, _adapter_snapshot
from rl.phase11.selection import adapter_files
from rl.phase12 import config
from rl.phase12.common import append, fingerprint, read, rows, write
from rl.phase12.demonstrations import masked_examples, validate
from rl.phase12.diagnostics import inspect_completion
from rl.phase12.metrics import reward_groups
from rl.phase12.sampling import SEED_POLICY, bind_request_seeds


def tool_schemas() -> list[dict[str, Any]]:
    from transformers.utils import get_json_schema

    from rl.phase12.environment import ComparisonEnv

    return [
        get_json_schema(getattr(ComparisonEnv, name))
        # GRPOTrainer extracts methods with inspect.getmembers (alphabetical order).
        for name in sorted(("get_invoice", "refund_invoice", "escalate_dispute"))
    ]


def trainer_class() -> type:
    from trl import GRPOTrainer

    class InstrumentedTrainer(GRPOTrainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if not self.use_vllm or self.args.vllm_mode != "colocate":
                raise ValueError("Phase 12 execution requires colocated vLLM")
            bind_request_seeds(self.vllm_generation.llm, self.args.seed)

        def _generate(self, prompts: list) -> Any:
            result = super()._generate(prompts)
            ids, masks, messages = result[1], result[2], result[3]
            stop = {self._tokenizer.eos_token_id, self._tokenizer.pad_token_id}
            for index, env in enumerate(self.environments):
                calls = [
                    {"name": o["name"], "arguments": o["arguments"]}
                    for o in env._phase12_observations
                ]
                env._phase12_generation = inspect_completion(
                    ids[index],
                    messages[index],
                    self._tokenizer.decode(ids[index]),
                    stop,
                    self.max_completion_length,
                    calls,
                )
                env._phase12_generation["generated_tokens"] = (
                    sum(masks[index]) if masks is not None else len(ids[index])
                )
            return result

    return InstrumentedTrainer


def reward_capture(output: Path, metadata: dict[str, Any]) -> Any:
    attempts: Counter[str] = Counter()

    def exact_reward(
        *, environments: list[Any], completions: list[Any], task_id: list[str], **_: Any
    ) -> list[float]:
        rewards = []
        for env, completion, key in zip(environments, completions, task_id, strict=True):
            if key not in metadata:
                raise ValueError("unknown task in reward callback")
            reward = float(env.reward)
            if reward not in (0.0, 1.0) or not env._phase12_generation:
                raise ValueError("nonbinary reward or missing generation diagnostics")
            evidence = env._baseline_evidence()
            observed = env._phase12_executed_calls()
            if evidence["tool_calls"] != observed:
                write(
                    output.with_name("evidence-mismatch.json"),
                    {
                        "task_id": key,
                        "expected_tool_calls": observed,
                        "recorded_tool_calls": evidence["tool_calls"],
                        "observations": env._phase12_observations,
                        "generation": env._phase12_generation,
                        "verifier_diagnostics": evidence["verifier_diagnostics"],
                    },
                )
                raise ValueError("missing or inconsistent executed-tool evidence")
            append(
                output,
                {
                    **metadata[key],
                    **evidence,
                    **env._phase12_generation,
                    "task_id": key,
                    "attempt": attempts[key],
                    "reward": reward,
                    "completion": completion,
                    "observations": env._phase12_observations,
                },
            )
            attempts[key] += 1
            rewards.append(reward)
        return rewards

    return exact_reward


def dataset(root: Path, selected: list[str]) -> tuple[Any, Any, dict[str, Any]]:
    from datasets import Dataset
    from trl.experimental.harbor import HarborSpec

    from rl.phase12.environment import ComparisonEnv

    metadata = read(root / "metadata.json")
    if not set(selected) <= set(metadata) or len(selected) != len(set(selected)):
        raise ValueError("unknown or duplicate task IDs")
    spec = HarborSpec(str(root / "execution"), agent=ComparisonEnv, environment_type="docker")
    values = [
        {**row, "task_id": Path(row["task_dir"]).name}
        for row in spec.train_dataset
        if Path(row["task_dir"]).name in selected
    ]
    if len(values) != len(selected):
        raise ValueError("executable dataset does not cover requested tasks")
    return Dataset.from_list(values), spec, {key: metadata[key] for key in selected}


def close_environments(trainer: Any) -> None:
    for env in trainer.environments or []:
        env._run(env._stop())


def evaluate(
    root: Path,
    selected: list[str],
    output: Path,
    exp: dict[str, Any],
    budget: int,
    seed: int,
    adapter: Path | None = None,
    thinking: bool = False,
) -> None:
    from transformers import set_seed

    set_seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    before = adapter_files(adapter) if adapter else None
    token = config.tokenizer(exp)
    data, spec, metadata = dataset(root, selected)
    settings = config.grpo_config(exp, output, budget, seed, training=False, thinking=thinking)
    trainer = trainer_class()(
        model=config.model(exp, adapter),
        processing_class=token,
        args=settings,
        eval_dataset=data,
        environment_factory=spec.environment_factory,
        reward_funcs=reward_capture(output / "rollouts.jsonl", metadata),
        **({"peft_config": config.lora(exp)} if adapter is None else {}),
    )
    started = time.perf_counter()
    try:
        metrics = trainer.evaluate()
        if (
            trainer.state.global_step != 0
            or trainer.optimizer is not None
            or list(output.glob("checkpoint-*"))
        ):
            raise ValueError("evaluation changed training state")
        if adapter and adapter_files(adapter) != before:
            raise ValueError("evaluation modified adapter")
        write(
            output / "result.json",
            {
                "execution_valid": True,
                "metrics": metrics,
                "global_step": 0,
                "optimizer_created": False,
                "training_performed": False,
                "adapter_files": before,
                "elapsed_seconds": time.perf_counter() - started,
                "inference_sha256": config.inference_hash(exp, budget, thinking),
                "seed": seed,
                "sampling_seed_policy": SEED_POLICY,
            },
        )
    finally:
        close_environments(trainer)


def train(
    root: Path, demos: Path, output: Path, exp: dict[str, Any], budget: int, method: str, seed: int
) -> None:
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed

    set_seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    token = config.tokenizer(exp)
    data, spec, metadata = dataset(root, sorted(read(root / "metadata.json")))
    if len(metadata) != 160:
        raise ValueError("training requires exactly 160 tasks")

    class Metrics(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> None:
            if logs:
                append(
                    output / "training-metrics.jsonl",
                    {
                        "step": state.global_step,
                        **{
                            k: float(v)
                            for k, v in logs.items()
                            if isinstance(v, (int, float)) and math.isfinite(v)
                        },
                    },
                )

    extra: dict[str, Any] = {}
    if method == "grpo":
        settings = config.grpo_config(exp, output / "checkpoints", budget, seed, training=True)
        cls = trainer_class()
        extra = {
            "environment_factory": spec.environment_factory,
            "reward_funcs": reward_capture(output / "rollouts.jsonl", metadata),
        }
    elif method == "sft":
        from trl import SFTTrainer
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

        records = rows(demos / "demonstrations.jsonl")
        if (
            len(records) != 160
            or {r["task_id"] for r in records} != set(metadata)
            or fingerprint(records) != read(demos / "manifest.json")["records_sha256"]
        ):
            raise ValueError("SFT demonstrations differ from verified train coverage")
        examples = []
        for record in records:
            validate(record, set(metadata))
            examples.extend(
                masked_examples(record, token, tool_schemas(), exp["sft"]["max_length"])
            )
        data = Dataset.from_list(examples)
        settings = config.sft_config(exp, output / "checkpoints", seed)
        cls = SFTTrainer
        extra = {"data_collator": DataCollatorForLanguageModeling(pad_token_id=token.pad_token_id)}
        write(
            output / "sft-data.json",
            {
                "demonstrations": 160,
                "assistant_turn_examples": len(examples),
                "supervised_tokens_per_epoch": sum(
                    sum(i != -100 for i in e["labels"]) for e in examples
                ),
                "input_tokens_per_epoch": sum(len(e["input_ids"]) for e in examples),
            },
        )
    else:
        raise ValueError("unknown training method")
    write(output / "effective-config.json", settings.to_dict())
    trainer = cls(
        model=config.model(exp),
        processing_class=token,
        args=settings,
        train_dataset=data,
        peft_config=config.lora(exp),
        callbacks=[Metrics()],
        **extra,
    )
    if trainer.accelerator.num_processes != 1:
        raise ValueError("canonical batch and exposure budgets require one GPU process")
    before = _adapter_snapshot(trainer.model)
    started = time.perf_counter()
    try:
        result = trainer.train()
        delta = _adapter_delta(before, _adapter_snapshot(trainer.model))
        if delta["delta_l2_norm"] <= 0 or not math.isfinite(delta["delta_l2_norm"]):
            raise ValueError("adapter did not change finitely")
        write(output / "adapter-delta.json", delta)
        if method == "grpo":
            if trainer.state.global_step != 320:
                raise ValueError("incomplete GRPO steps")
            write(output / "reward-groups.json", reward_groups(rows(output / "rollouts.jsonl")))
        elif abs(float(trainer.state.epoch) - 3.0) > 1e-6:
            raise ValueError("incomplete SFT epochs")
        checkpoints = sorted(
            (output / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
        )
        if len(checkpoints) != (2 if method == "grpo" else 3):
            raise ValueError("missing canonical checkpoints")
        import shutil

        candidates = {}
        for checkpoint in checkpoints:
            step = int(checkpoint.name.split("-")[-1])
            target = output / "adapters" / str(step)
            target.mkdir(parents=True)
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                shutil.copy2(checkpoint / name, target / name)
            candidates[str(step)] = adapter_files(target)
        if method == "grpo":
            captured = rows(output / "rollouts.jsonl")
            token_evidence = {
                "environment_rollouts": len(captured),
                "generated_tokens": sum(r["generated_tokens"] for r in captured),
                "conversation_completion_tokens": sum(
                    r["completion_token_count"] for r in captured
                ),
            }
        else:
            evidence = read(output / "sft-data.json")
            token_evidence = {
                "environment_rollouts_during_training": 0,
                "input_tokens": evidence["input_tokens_per_epoch"] * 3,
                "supervised_tokens": evidence["supervised_tokens_per_epoch"] * 3,
            }
        write(
            output / "training-result.json",
            {
                "execution_valid": True,
                "method": method,
                "seed": seed,
                "global_step": trainer.state.global_step,
                "metrics": result.metrics,
                "adapters": candidates,
                "elapsed_seconds": time.perf_counter() - started,
                "inference_sha256": config.inference_hash(exp, budget),
                "token_budget_evidence": token_evidence,
                "sampling_seed_policy": SEED_POLICY if method == "grpo" else None,
            },
        )
    finally:
        if method == "grpo":
            close_environments(trainer)
