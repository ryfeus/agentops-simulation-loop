"""One real Bedrock structured decision per frozen replay row; no invented confidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


class Decision(BaseModel):
    action: Literal["refund", "escalate", "no_action", "defer"]


def messages_for(row: dict[str, Any]) -> list[dict[str, str]]:
    """Only the same instructions, options and serialized observed state reach Bedrock."""
    return [
        {
            "role": "system",
            "content": row["instructions"]
            + "\nAvailable choices: "
            + json.dumps(row["options"], ensure_ascii=False)
            + "\nReturn the single selected action using the supplied Decision schema.",
        },
        {
            "role": "user",
            "content": row["state"]
            if isinstance(row["state"], str)
            else json.dumps(row["state"], sort_keys=True, separators=(",", ":")),
        },
    ]


async def run(args: argparse.Namespace) -> None:
    from agentops_demo.agent.model import create_model
    from agentops_demo.contracts.agent_config import AgentConfig

    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError("output must be fresh; replay is not silently retried")
    records = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    records = [row for row in records if row["split"] == args.split]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError("no matching rows")
    if len(records) > 500:
        raise ValueError("replay ceiling is 500 calls")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = AgentConfig.model_validate(
        {
            "agent": {"source_revision": source},
            "model": {"provider": "bedrock", "model_id": args.model},
            "prompt": {"version": "billing-v2"},
            "tools": {"version": "billing-mcp-v1"},
            "harness": {"framework": "langgraph", "version": "v1"},
        }
    )
    os.environ["BILLING_MODEL_TEMPERATURE"] = "0"
    os.environ["BILLING_MODEL_MAX_TOKENS"] = "128"
    os.environ["BILLING_MODEL_TOTAL_ATTEMPTS"] = "1"
    started = time.perf_counter()
    from scripts.aws_context import verified_session

    session = verified_session()
    model = create_model(config, region_name="us-west-2", aws_session=session)
    structured = model.with_structured_output(Decision, include_raw=True)
    initialization_ms = (time.perf_counter() - started) * 1000
    manifest = {
        "variant": "same-observed-state-bedrock-decision-replay-v1",
        "model": args.model,
        "region": "us-west-2",
        "calls_planned": len(records),
        "concurrency": 2,
        "attempts_per_row": 1,
        "sdk_total_attempts": 1,
        "timeout_seconds": 60,
        "max_output_tokens": 128,
        "temperature": 0,
        "source_revision": source,
        "split": args.split,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "initialization_ms": initialization_ms,
        "confidence": "unavailable; no probabilities or calibration inferred from generated text",
        "limitation": "Exploratory single-call structured replay; not free-form agent evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    semaphore = asyncio.Semaphore(2)
    wall_started = time.perf_counter()
    completed: list[dict[str, Any]] = []

    async def predict(index: int, row: dict[str, Any]) -> None:
        async with semaphore:
            output = {
                key: row[key]
                for key in ("example_id", "group_id", "split", "label_action", "prohibited_actions")
                if key in row
            }
            output.update(
                action=None,
                error=None,
                probabilities={},
                confidence=None,
                model=args.model,
                usage={},
                call_index=index,
                warm=index >= 2,
            )
            before = time.perf_counter()
            try:
                result = await asyncio.wait_for(structured.ainvoke(messages_for(row)), timeout=60)
                raw = result.get("raw")
                output["usage"] = getattr(raw, "usage_metadata", None) or {}
                output["response_metadata"] = getattr(raw, "response_metadata", None) or {}
                if result.get("parsing_error") or result.get("parsed") is None:
                    raise ValueError(
                        f"structured output parse failed: {result.get('parsing_error')}"
                    )
                output["action"] = result["parsed"].action
            except Exception as exc:
                output["error"] = f"{type(exc).__name__}: {exc}"
            output["latency_ms"] = (time.perf_counter() - before) * 1000
            output["input_tokens"] = output["usage"].get("input_tokens")
            output["output_tokens"] = output["usage"].get("output_tokens")
            completed.append(output)
            with args.output.open("a") as handle:
                handle.write(json.dumps(output, sort_keys=True) + "\n")
            if len(completed) % 10 == 0 or len(completed) == len(records):
                print(
                    json.dumps(
                        {
                            "completed": len(completed),
                            "total": len(records),
                            "errors": sum(item["error"] is not None for item in completed),
                        }
                    ),
                    flush=True,
                )

    await asyncio.gather(*(predict(index, row) for index, row in enumerate(records)))
    manifest.update(
        wall_ms=(time.perf_counter() - wall_started) * 1000,
        calls_completed=len(completed),
        errors=sum(row["error"] is not None for row in completed),
        input_tokens=sum(row["input_tokens"] or 0 for row in completed),
        output_tokens=sum(row["output_tokens"] or 0 for row in completed),
        output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "dev", "calibration"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="us.anthropic.claude-sonnet-4-6")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
