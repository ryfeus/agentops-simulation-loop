# ADR 0005: AgentCore Runtime and Aurora DSQL vertical slice

- Status: Accepted
- Date: 2026-09-08

## Context

Phase 2 proves the candidate and verifier path in Harbor. The first production
slice must prove that the same agent, prompt, LangGraph harness, and MCP schemas
can execute on AWS without adding UI or evaluation infrastructure.

## Decisions

1. AgentCore Runtime is the production runtime and implements the HTTP
   `/ping` and `/invocations` contract on port 8080.
2. The agent and Billing MCP server are co-located in one Phase 3 container, but
   communicate through real localhost Streamable HTTP.
3. Aurora DSQL implements the production `BillingRepository`; SQLite remains the
   local and Harbor repository.
4. DSQL connections use IAM authentication. The runtime uses a custom
   `billing_runtime` database role and `dsql:DbConnect`; admin access is limited
   to deployment bootstrap and reset operations.
5. Terraform owns DSQL, ECR, the independent execution role, and AgentCore
   Runtime in `us-west-2`. The operator must provide a 12-digit
   `EXPECTED_AWS_ACCOUNT_ID`; the credential wrapper compares it with STS before
   executing AWS commands. Terraform continues deriving account-aware ARNs from
   the actual caller identity. AgentCore creates and owns the runtime's `DEFAULT`
   endpoint as part of Runtime creation; Terraform exports its deterministic ARN
   and does not request a duplicate endpoint.
6. The runtime uses HTTP protocol and PUBLIC networking for this demo. Private
   networking is deferred.
7. The Python 3.12 container is built for ARM64 and deployed only through an
   immutable ECR digest.
8. Production provenance binds the exact Git revision, wheel SHA-256,
   AgentConfig fingerprint, and image digest before Terraform creates or updates
   the runtime.
9. The first acceptance call uses `InvokeAgentRuntime` directly and requires a
   completed `get_invoice` through LangGraph, MCP, and DSQL.
10. Foundation plans and applies preserve a deployed runtime by validating and
    reusing its generated deployment inputs. Partial or stale inputs fail before
    apply. Credential-free CI builds the complete production `linux/arm64` image
    with a cache-only Buildx output.
11. Runtime health is process-local: `/ping` reports `HealthyBusy` while any
    invocation is active and returns to `Healthy` after completion, failure, or
    cancellation. No persistent session or health store is introduced.

## Consequences

Production and Harbor now share the candidate boundary while differing only at
the runtime and repository/world boundary. A passing smoke establishes service
connectivity and provenance, not model or prompt quality. Assistant UI, AG-UI,
Lambda streaming, AgentCore Online Evaluations, trace conversion, private
networking, multi-region DSQL, persistent conversations, and EC2 Harbor workers
remain deferred.
