"""Executable tool demonstrations and explicit assistant-turn token masking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentops_demo.contracts.scenario import Scenario
from rl.phase12.common import append, fingerprint, rows, write


def teacher_calls(scenario: Scenario) -> list[dict[str, Any]]:
    """A deterministic teacher plan, executed through tools, never through Oracle SQL."""
    assert scenario.benchmark
    calls = [
        {"name": "get_invoice", "arguments": {"invoice_id": invoice}}
        for invoice in scenario.benchmark.trajectory.required_inspections
    ]
    for invariant in scenario.expected_invariants:
        if invariant.type in {"must_refund", "must_escalate"}:
            calls.append(
                {
                    "name": "refund_invoice"
                    if invariant.type == "must_refund"
                    else "escalate_dispute",
                    "arguments": {
                        "invoice_id": invariant.invoice_id,
                        "reason": "Customer requested billing resolution",
                    },
                }
            )
    return calls


def conversation(environment: Any, scenario: Scenario, task: Path) -> dict[str, Any]:
    observation = environment.reset(task_dir=str(task))
    messages: list[dict[str, Any]] = [{"role": "user", "content": observation}]
    outcomes = []
    for call in teacher_calls(scenario):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": call}],
            }
        )
        result = getattr(environment, call["name"])(**call["arguments"])
        decoded = json.loads(result)
        if decoded.get("ok") is False and not (
            call["name"] == "get_invoice" and decoded.get("error_type") == "InvoiceNotFoundError"
        ):
            raise ValueError("teacher received an unexpected tool error")
        messages.append({"role": "tool", "name": call["name"], "content": result})
        invoice = call["arguments"]["invoice_id"]
        if decoded.get("ok") is False:
            outcomes.append(f"{invoice}: not found; no action taken.")
        elif call["name"] == "get_invoice":
            outcomes.append(f"{invoice}: observed {decoded['invoice']['status']}.")
        else:
            action = "refunded" if call["name"] == "refund_invoice" else "escalated for review"
            outcomes.append(f"{invoice}: {action}.")
    messages.append({"role": "assistant", "content": " ".join(outcomes)})
    reward = float(environment.reward)
    if reward != 1.0:
        raise ValueError(f"teacher demonstration failed verifier: {scenario.id}")
    evidence = environment._baseline_evidence()
    if evidence["tool_calls"] != teacher_calls(scenario):
        raise ValueError("executed teacher trajectory differs from planned calls")
    return {
        "task_id": scenario.id,
        "messages": messages,
        "reward": reward,
        "tool_calls": evidence["tool_calls"],
        "verifier_components": evidence["verifier_components"],
        "source": "executed-billing-tools",
        "complete": True,
        "truncated": False,
    }


def validate(record: dict[str, Any], allowed: set[str]) -> None:
    if (
        record.get("task_id") not in allowed
        or record.get("reward") != 1.0
        or record.get("source") != "executed-billing-tools"
        or record.get("complete") is not True
        or record.get("truncated") is not False
    ):
        raise ValueError("unverified, truncated, or non-training demonstration")
    messages = record["messages"]
    if (
        messages[0]["role"] != "user"
        or messages[-1]["role"] != "assistant"
        or not messages[-1].get("content")
    ):
        raise ValueError("demonstration lacks a complete conversation")
    calls = []
    for i in range(1, len(messages) - 1, 2):
        assistant, tool = messages[i : i + 2]
        if assistant["role"] != "assistant" or tool["role"] != "tool":
            raise ValueError("unpaired assistant/tool messages")
        function = assistant["tool_calls"][0]["function"]
        if tool["name"] != function["name"]:
            raise ValueError("tool observation belongs to a different call")
        json.loads(tool["content"])
        calls.append(function)
    if not calls or calls != record["tool_calls"]:
        raise ValueError("demonstration evidence differs from its conversation")


def masked_examples(
    record: dict[str, Any], token: Any, tools: list[dict[str, Any]], max_length: int
) -> list[dict[str, Any]]:
    """One example per assistant turn; all preceding user/tool/assistant tokens are masked.

    Prefix token equality is checked rather than relying on substring searches or an
    upstream template's optional generation masks. This also teaches every tool call.
    """
    messages = record["messages"]
    examples = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        # Transformers 5 defaults to BatchEncoding; prefix slicing and loss masks
        # require the flat token IDs used by the native generation contract.
        kwargs = {
            "tools": tools,
            "tokenize": True,
            "return_dict": False,
            "enable_thinking": False,
        }
        prefix = token.apply_chat_template(messages[:index], add_generation_prompt=True, **kwargs)
        complete = token.apply_chat_template(
            messages[: index + 1], add_generation_prompt=False, **kwargs
        )
        if complete[: len(prefix)] != prefix or len(complete) <= len(prefix):
            raise ValueError("chat template does not preserve the assistant generation prefix")
        if len(complete) > max_length:
            raise ValueError("SFT demonstration would be truncated")
        examples.append(
            {
                "input_ids": complete,
                "attention_mask": [1] * len(complete),
                "labels": [-100] * len(prefix) + complete[len(prefix) :],
            }
        )
    return examples


def collect(dataset: Path, output: Path, metadata: dict[str, Any]) -> None:
    from agentops_demo.validation.scenario import load_scenario
    from rl.phase12.environment import ComparisonEnv

    output.mkdir(parents=True, exist_ok=False)
    env = ComparisonEnv(environment_type="docker")
    try:
        for task_id in sorted(metadata):
            task = dataset / "tasks" / task_id
            scenario = load_scenario(task / "scenario.yaml")
            record = conversation(env, scenario, task)
            validate(record, set(metadata))
            append(output / "demonstrations.jsonl", record)
    finally:
        env._run(env._stop())
    records = rows(output / "demonstrations.jsonl")
    write(
        output / "manifest.json",
        {
            "count": len(records),
            "task_ids": sorted(metadata),
            "records_sha256": fingerprint(records),
            "execution_valid": True,
        },
    )
