from __future__ import annotations

import pytest

from agentops_demo.agentcore.candidates import (
    CandidateRuntimePool,
    SessionCandidateBindings,
    SessionCandidateConflictError,
    resolve_candidate,
)


def test_candidate_resolver_preserves_default_and_derives_scripted_configs(agent_config) -> None:
    bedrock = resolve_candidate(agent_config, "bedrock")
    bad = resolve_candidate(agent_config, "scripted-bad")
    correct = resolve_candidate(agent_config, "scripted-correct")

    assert resolve_candidate(agent_config, None) is agent_config
    assert bedrock is agent_config
    assert (bad.model.provider, bad.model.model_id) == ("scripted", "bad")
    assert (correct.model.provider, correct.model.model_id) == ("scripted", "correct")
    assert bad.agent == correct.agent == agent_config.agent
    assert bad.prompt == correct.prompt == agent_config.prompt
    assert len({bedrock.fingerprint(), bad.fingerprint(), correct.fingerprint()}) == 3


@pytest.mark.asyncio
async def test_runtime_pool_reuses_by_fingerprint_and_recovers_after_failure(agent_config) -> None:
    calls: list[str] = []

    async def factory(*, config, mcp_url):  # type: ignore[no-untyped-def]
        del mcp_url
        calls.append(config.model.model_id)
        if config.model.model_id == "bad" and calls.count("bad") == 1:
            raise RuntimeError("transient")
        return object()

    pool = CandidateRuntimePool(mcp_url="http://mcp", runtime_factory=factory)  # type: ignore[arg-type]
    bad = resolve_candidate(agent_config, "scripted-bad")
    correct = resolve_candidate(agent_config, "scripted-correct")

    with pytest.raises(RuntimeError, match="transient"):
        await pool.runtime_for(bad)
    correct_runtime = await pool.runtime_for(correct)
    assert await pool.runtime_for(correct) is correct_runtime
    bad_runtime = await pool.runtime_for(bad)
    assert bad_runtime is not correct_runtime
    assert calls == ["bad", "correct", "bad"]


@pytest.mark.asyncio
async def test_session_candidate_binding_rejects_a_different_fingerprint() -> None:
    bindings = SessionCandidateBindings()
    await bindings.bind_or_validate(session_id="session", fingerprint="a" * 64)
    await bindings.bind_or_validate(session_id="session", fingerprint="a" * 64)
    with pytest.raises(SessionCandidateConflictError, match="different candidate"):
        await bindings.bind_or_validate(session_id="session", fingerprint="b" * 64)
    await bindings.bind_or_validate(session_id="other", fingerprint="b" * 64)
