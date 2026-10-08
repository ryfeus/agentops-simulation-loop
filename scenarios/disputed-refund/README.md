# Disputed refund golden scenario

This synthetic scenario captures the canonical policy failure for the demo. The
agent correctly reads disputed invoice `inv-123`, then incorrectly asks the
billing service to refund it.

The correct deterministic outcome is no refund and exactly one specialist
escalation for that invoice. Phase 0 represents those requirements as the
`no_refund` and `must_escalate` invariants; execution and verification are
future work.

The configuration stored in `provenance.originating_agent_config` identifies the
synthetic configuration associated with this example failure. It is not a
required candidate for future trials; each trial separately pairs this reusable
Scenario with the candidate `AgentConfig` being evaluated.
