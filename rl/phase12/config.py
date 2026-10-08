"""One inference contract for both methods, diagnostics, and the frozen baseline."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from rl.phase12.common import fingerprint, read

MANIFEST = Path(__file__).with_name("experiment.json")


def load(path: Path = MANIFEST) -> dict[str, Any]:
    value = read(path)
    # v1 is immutable; editing the canonical file intentionally creates a new source identity.
    if value != read(MANIFEST):
        raise ValueError("experiment differs from the canonical Phase 12 manifest")
    if value["schema_version"] != "1" or value["experiment_id"] != "billing-phase12-v1":
        raise ValueError("unsupported experiment")
    for key in ("revision", "tokenizer_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", value["model"][key]):
            raise ValueError("model and tokenizer require immutable commit revisions")
    if not re.fullmatch(r"[0-9a-f]{64}", value["model"]["chat_template_sha256"]):
        raise ValueError("missing template hash")
    if value["seeds"] != [42, 43, 44] or value["attempts"] != 8:
        raise ValueError("canonical seed/evaluation schedule changed")
    return value


def tokenizer(exp: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    token = AutoTokenizer.from_pretrained(
        exp["model"]["id"], revision=exp["model"]["tokenizer_revision"]
    )
    if (
        hashlib.sha256(token.chat_template.encode()).hexdigest()
        != exp["model"]["chat_template_sha256"]
    ):
        raise ValueError("loaded tokenizer chat template differs from manifest")
    return token


def model(exp: dict[str, Any], adapter: Path | None = None) -> Any:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    # TRL's colocated vLLM opens model.name_or_path without forwarding a revision.
    # A resolved immutable snapshot keeps that second load pinned as well.
    snapshot = snapshot_download(exp["model"]["id"], revision=exp["model"]["revision"])
    result = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, dtype="bfloat16")
    if adapter is not None:
        from peft import PeftModel

        result = PeftModel.from_pretrained(result, adapter, is_trainable=False)
    return result


def lora(exp: dict[str, Any]) -> Any:
    from peft import LoraConfig

    return LoraConfig(**exp["lora"])


def inference(exp: dict[str, Any], budget: int, thinking: bool = False) -> dict[str, Any]:
    if budget not in exp["rollout"]["budgets"]:
        raise ValueError("unregistered rollout budget")
    return {
        "max_completion_length": budget,
        "max_tool_calling_iterations": 4
        if thinking
        else exp["rollout"]["max_tool_calling_iterations"],
        "chat_template_kwargs": {"enable_thinking": thinking},
        **{
            key: exp["rollout"][key]
            for key in ("temperature", "top_p", "top_k", "repetition_penalty")
        },
        "min_p": None,
    }


def grpo_config(
    exp: dict[str, Any],
    output: Path,
    budget: int,
    seed: int,
    *,
    training: bool,
    thinking: bool = False,
    cpu: bool = False,
) -> Any:
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=str(output),
        seed=seed,
        data_seed=seed,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=4,
        num_generations_eval=4,
        max_steps=exp["grpo"]["steps"] if training else 1,
        learning_rate=exp["grpo"]["learning_rate"],
        save_strategy="steps" if training else "no",
        save_steps=160,
        save_total_limit=2,
        logging_steps=1,
        report_to="none",
        push_to_hub=False,
        bf16=not cpu,
        use_cpu=cpu,
        do_eval=not training,
        use_vllm=not cpu,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.30,
        beta=0.0,
        scale_rewards="group",
        loss_type="dapo",
        mask_truncated_completions=False,
        **inference(exp, budget, thinking),
    )


def sft_config(exp: dict[str, Any], output: Path, seed: int, *, cpu: bool = False) -> Any:
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(output),
        seed=seed,
        data_seed=seed,
        num_train_epochs=exp["sft"]["epochs"],
        learning_rate=exp["sft"]["learning_rate"],
        per_device_train_batch_size=exp["sft"]["batch_size"],
        gradient_accumulation_steps=exp["sft"]["gradient_accumulation_steps"],
        max_length=exp["sft"]["max_length"],
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=1,
        report_to="none",
        push_to_hub=False,
        bf16=not cpu,
        use_cpu=cpu,
    )


def inference_hash(exp: dict[str, Any], budget: int, thinking: bool = False) -> str:
    from rl.phase12.sampling import SEED_POLICY

    return fingerprint(
        {
            "model": exp["model"],
            "settings": inference(exp, budget, thinking),
            "sampling_seed_policy": SEED_POLICY,
        }
    )
