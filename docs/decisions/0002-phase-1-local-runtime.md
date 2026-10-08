# ADR 0002: Phase 1 local runtime

- Status: Accepted
- Date: 2026-09-07

## Context

Phase 1 must execute the Phase 0 contracts locally, reproduce the known bad
decision deterministically, and remain reusable by future Harbor and AgentCore
environments.

## Decisions

1. The agent discovers its three domain tools only through MCP Streamable HTTP.
2. `BillingService` owns policy; `SQLiteBillingRepository` owns persistence.
3. Permissive policy is the local default so the bad disputed-refund trajectory
   remains reproducible; enforced policy is an explicit alternative.
4. SQLite fixture state is replaced from `Scenario.initial_state` and amounts are
   stored as text.
5. Successful tools return Phase 0 models as structured content; failures use
   stable coded domain exceptions that remain readable across MCP.
6. Deterministic model responses still execute the compiled LangGraph and real
   MCP transport.
7. Graph construction remains an asynchronous factory. A module-level graph and
   `langgraph.json` are deferred because model and MCP discovery require runtime
   configuration.
8. Bedrock support is optional and excluded from normal tests and CI.
9. FastMCP 3.x is used until LangChain's MCP adapter supports the MCP SDK 2.x
   required by FastMCP 4.x.
10. Graph construction takes the complete candidate `AgentConfig`, uses its
    prompt version, and rejects unsupported tool and harness identities.
11. SQLite enforces one refund and one escalation per invoice with unique
    indexes in addition to repository-level duplicate checks.

## Consequences

The local system is executable without AWS, Docker, or an external model. Future
environments can replace the repository and supply a different model without
changing agent logic or MCP schemas. Harbor, AgentCore, observability, trace
conversion, production auth, and deployment remain out of scope.
