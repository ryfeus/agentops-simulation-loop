# ADR 0001: Phase 0 system boundaries

- Status: Accepted
- Date: 2026-09-07

## Context

The demo must turn production policy failures into deterministic simulations
without coupling the agent to a production database, observability vendor payload,
or benchmark task format.

## Decisions

1. LangGraph will be the common future agent harness.
2. MCP is the stable, domain-oriented tool boundary.
3. Production and simulation use identical MCP schemas.
4. Production and simulation may use different repository implementations.
5. SQLite is the intended first simulation backend.
6. `Scenario` is the canonical intermediate representation.
7. Harbor-specific task and verifier generation happens downstream of Scenario.
8. All demo fixtures use synthetic data only.
9. End-user authentication is not a Phase 0 concern.
10. Candidate identity versions the source, model, prompt, tools, and harness—not
    only the model.
11. Scenario provenance may retain an optional originating configuration, while a
    candidate configuration remains separate and is paired with the Scenario at
    trial time.
12. Source provenance is provider-neutral and supports synthetic or trace-derived
    scenarios, with optional structured evaluation evidence.
13. Billing entity IDs are unique per entity type rather than globally.
14. `no_refund` requires zero final refunds and `must_escalate` requires exactly
    one final escalation for the referenced invoice.
15. Phase 0 records observed tool calls but intentionally defers normalized tool
    results and state transitions until real AgentCore/LangGraph traces are
    inspected.

## Consequences

Phase 0 contains schemas, fixtures, validation, tests, and documentation only.
LangGraph execution, FastMCP, AWS SDKs, AgentCore, telemetry, databases, Harbor,
containers, infrastructure, authentication, and trace conversion are deferred.
