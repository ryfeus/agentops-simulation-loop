# ADR 0003: AWS target architecture

- Status: Accepted
- Date: 2026-09-07

## Context

Phase 1 is a local vertical slice. Future phases need a shared target for AWS
deployment and Harbor evaluation without changing the candidate agent identity
or its MCP contract between production and simulation.

## Decisions

1. Terraform is the primary AWS provisioning path. CDK, Pulumi, and manual
   console configuration are not project defaults.
2. Amazon Aurora DSQL is the target database behind a
   `DSQLRepository`; SQLite remains the simulation default. A PostgreSQL sidecar
   is reserved for tasks that specifically require PostgreSQL semantics.
3. The UI uses assistant-ui and communicates over AG-UI.
4. The production agent runs LangGraph on Amazon Bedrock AgentCore Runtime and
   reaches billing tools through MCP.
5. Harbor runs simulations locally and on EC2 workers in AWS.
6. The online detection loop uses AgentCore Observability, CloudWatch traces,
   and AgentCore Online Evaluations. Failed traces are normalized into reusable
   Scenarios before Harbor execution.

```text
assistant-ui -> AG-UI -> AgentCore Runtime -> LangGraph -> MCP
                                                       -> Billing MCP
                                                       -> DSQLRepository
                                                       -> Aurora DSQL

AgentCore Runtime -> AgentCore Observability / CloudWatch
                  -> Online Evaluations -> failed trace -> Scenario -> Harbor

Harbor (local or EC2 workers) -> LangGraph -> MCP -> Billing MCP
                                         -> SQLiteRepository -> SQLite
```

The parity boundary is:

| Concern | Production | Simulation |
| --- | --- | --- |
| Agent code | Same | Same |
| Candidate `AgentConfig` | Same | Same |
| Prompt and model | Same | Same |
| MCP tool schemas | Same | Same |
| Harness | Same | Same |
| Repository | DSQL | SQLite |
| World state | Live/demo AWS | Frozen Scenario |
| Runtime | AgentCore | Harbor |

Only the world and runtime boundary should change; the candidate configuration
being evaluated should not.

## Reuse policy

Reuse reviewed Terraform, DSQL, AgentCore, AG-UI and assistant-ui patterns where appropriate, adapting them to the contracts in this repository. Earlier private source references are omitted from this export.

## Consequences

The target is decision-ready for later phases, but this ADR introduces no AWS,
UI, Harbor, observability, or deployment implementation in Phase 1.
