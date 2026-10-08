from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest
from opentelemetry import baggage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentops_demo.agentcore.tracing import candidate_span_attributes, invocation_trace_context


def test_candidate_attributes_are_complete_and_secret_free(agent_config) -> None:
    attributes = candidate_span_attributes(agent_config, "permissive")
    assert attributes["agentops.source_revision"] == "abc123"
    assert attributes["agentops.agent_config_fingerprint"] == agent_config.fingerprint()
    assert attributes["agentops.model_provider"] == agent_config.model.provider
    assert attributes["agentops.prompt_version"] == "billing-v1"
    assert set(attributes) == {
        "agentops.source_revision",
        "agentops.agent_config_fingerprint",
        "agentops.model_provider",
        "agentops.model_id",
        "agentops.prompt_version",
        "agentops.tools_version",
        "agentops.harness_framework",
        "agentops.harness_version",
        "agentops.billing_policy_mode",
    }


@pytest.mark.asyncio
async def test_trace_context_records_attributes_and_cleans_baggage(agent_config) -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("invocation"):
        async with invocation_trace_context(
            config=agent_config, policy_mode="permissive", session_id="session-123"
        ):
            assert baggage.get_baggage("session.id") == "session-123"
    assert baggage.get_baggage("session.id") is None
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes["session.id"] == "session-123"
    assert attributes["agentops.agent_config_fingerprint"] == agent_config.fingerprint()


@pytest.mark.asyncio
async def test_trace_context_cleans_baggage_after_failure(agent_config) -> None:
    with pytest.raises(RuntimeError):
        async with invocation_trace_context(
            config=agent_config, policy_mode="permissive", session_id="session-failed"
        ):
            raise RuntimeError("boom")
    assert baggage.get_baggage("session.id") is None


@pytest.mark.asyncio
async def test_trace_context_cleans_baggage_after_cancellation(agent_config) -> None:
    started = asyncio.Event()

    async def run() -> None:
        async with invocation_trace_context(
            config=agent_config, policy_mode="permissive", session_id="session-cancelled"
        ):
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await started.wait()
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    assert baggage.get_baggage("session.id") is None
