"""Safe per-invocation trace correlation and candidate provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from opentelemetry import baggage, context, trace

from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.contracts.scenario import BenchmarkMetadata
from agentops_demo.contracts.world_snapshot import (
    BillingWorldSnapshot,
    canonical_snapshot_json,
    snapshot_sha256,
)


class WorldSnapshotError(ValueError):
    """Raised when trace evidence cannot represent a complete snapshot."""


def world_snapshot_attributes(
    snapshot: BillingWorldSnapshot, *, maximum_bytes: int
) -> dict[str, str]:
    serialized = canonical_snapshot_json(snapshot)
    size = len(serialized.encode("utf-8"))
    if size > maximum_bytes:
        raise WorldSnapshotError(
            f"world snapshot is {size} bytes, exceeding the {maximum_bytes}-byte limit"
        )
    return {
        "agentops.world_state.before": serialized,
        "agentops.world_snapshot.schema_version": snapshot.schema_version,
        "agentops.world_snapshot.kind": "billing-demo",
        "agentops.world_snapshot.sha256": snapshot_sha256(snapshot),
    }


def candidate_span_attributes(config: AgentConfig, policy_mode: str) -> dict[str, str]:
    """Return the secret-free candidate identity recorded on an invocation span."""

    return {
        "agentops.source_revision": config.agent.source_revision,
        "agentops.agent_config_fingerprint": config.fingerprint(),
        "agentops.model_provider": config.model.provider,
        "agentops.model_id": config.model.model_id,
        "agentops.prompt_version": config.prompt.version,
        "agentops.tools_version": config.tools.version,
        "agentops.harness_framework": config.harness.framework,
        "agentops.harness_version": config.harness.version,
        "agentops.billing_policy_mode": policy_mode,
    }


def benchmark_context_attributes(context_value: BenchmarkMetadata) -> dict[str, str]:
    serialized = json.dumps(
        context_value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return {
        "agentops.benchmark_context": serialized,
        "agentops.benchmark_context.sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }


@asynccontextmanager
async def invocation_trace_context(
    *,
    config: AgentConfig,
    policy_mode: str,
    session_id: str,
    instruction: str | None = None,
    world_snapshot: BillingWorldSnapshot | None = None,
    world_snapshot_max_bytes: int = 32_768,
    benchmark_context: BenchmarkMetadata | None = None,
) -> AsyncIterator[None]:
    """Attach session baggage and annotate the active server span for one invocation."""

    span = trace.get_current_span()
    if span.is_recording():
        span.set_attributes(candidate_span_attributes(config, policy_mode))
        span.set_attribute("session.id", session_id)
        if instruction is not None:
            span.set_attribute("agentops.invocation.instruction", instruction)
        if world_snapshot is not None:
            span.set_attributes(
                world_snapshot_attributes(world_snapshot, maximum_bytes=world_snapshot_max_bytes)
            )
        if benchmark_context is not None:
            span.set_attributes(benchmark_context_attributes(benchmark_context))
    token = context.attach(baggage.set_baggage("session.id", session_id))
    try:
        yield
    finally:
        context.detach(token)
