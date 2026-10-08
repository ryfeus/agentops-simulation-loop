# Architecture

Phase 0 defines stable contracts, Phase 1 implements the local runtime path,
Phase 2 places it inside the first deterministic Harbor benchmark, Phase 3
deploys the production-shaped AgentCore and Aurora DSQL path, and Phase 4 adds
production-shaped trace detection without changing the agent-visible boundary.
Phase 5 turns captured failures into validated generated benchmarks. Phase 6
moves only the Harbor trial host onto ephemeral EC2; it does not add network
access to the benchmark or modify its semantics. Phase 7 joins a frozen live
failure to local calibration and EC2 evidence without adding a promotion path.

## Phase 6 EC2 execution plane

```text
guarded local controller (AWS + local SSH key)
       | Harbor 0.22 EC2 JobConfig
       v
an ephemeral public Ubuntu 24.04 agent environment
       | Docker bootstrap
       v
agent container (no network) <-> separately provisioned verifier environment
       |
       v
retained raw job + candidate-gate.json, then worker deletion
```

Terraform's optional `modules/harbor_ec2` foundation owns a public VPC/subnet,
IGW route, SSH security group, public key registration, and an AMI lookup. It
does not create VMs, private keys, NAT, instance profiles, production routing,
or cloud credentials for task containers. The controller identifies every
worker with project, phase, and run tags, and cleanup refuses broad selection.
Normal CI and all local Harbor execution remain AWS-free.

## Phase 7 evidence loop

```text
bounded Bedrock failure -> frozen trace + failure hashes -> taskify Scenario
          |                                                  |
          v                                                  v
  source identity retained                     local Harbor calibration (1/1/0)
                                                             |
                                                             v
                                               EC2 parity then four correct trials
                                                             |
                                                             v
                                               linked demo-report + zero workers
```

Bedrock is never replayed inside Harbor. For the supported disputed-refund
trajectory, taskify preserves the original Bedrock AgentConfig as source
provenance and derives a distinct `scripted/bad` calibration configuration only
for the isolated trajectory test. Each demo run verifies Scenario and rendered
task digests before EC2 work and stores only non-secret identifiers and hashes.

## Local Phase 1

```text
user or scripted model
         |
         v
   LangGraph agent
         |
         | MCP Streamable HTTP
         v
   Billing FastMCP
         |
         v
   BillingService
         |
         v
 SQLiteBillingRepository
         |
         v
    billing.db
```

The agent factory receives a complete candidate `AgentConfig` plus an injectable
model and discovers tools only through MCP. The configuration selects the prompt
and declares supported tool and harness versions, keeping the fingerprinted
identity aligned with runtime construction. The agent package remains independent
of the service, repository, FastMCP, and SQLite implementations.

The scripted model replaces only LLM decision-making. Its bad and correct modes
still exercise the compiled LangGraph, real MCP HTTP transport, billing service,
and SQLite repository. They prove the infrastructure and evaluation path, not
that prompt v2 improves a real model.

## One agent, two worlds

Production and simulation use the same agent configuration and
domain-oriented MCP schemas. Only the repository implementation and seeded data
change at the environment boundary.

```text
PRODUCTION (Phase 3)

InvokeAgentRuntime
  |
AgentCore Runtime HTTP adapter
  |
LangGraph agent
  | MCP
Billing MCP
  |
DSQLBillingRepository
  |
Aurora DSQL demo state
```

```text
SIMULATION (Phase 2)

Harbor
  |
LangGraph agent
  | MCP
Billing MCP
  |
SQLiteRepository
  |
seeded SQLite
```

Repository implementations remain behind the Phase 0 `BillingRepository`
protocol. Local and Harbor paths use SQLite; Phase 3 production uses DSQL with
IAM authentication. The agent sees only `get_invoice`,
`refund_invoice`, and `escalate_dispute`; it never sees SQL or storage details.

## Policy experiment

The service owns the disputed-invoice rule. Permissive mode deliberately allows
the baseline failure to be reproduced; enforced mode rejects that refund with
`disputed_invoice_requires_escalation`. The repository only persists state and
does not make policy decisions.

## Implemented AWS Phase 3

```text
InvokeAgentRuntime -> AgentCore Runtime -> LangGraph
    | MCP
    v
Billing MCP -> DSQLRepository -> Aurora DSQL

Harbor -> LangGraph -> MCP -> Billing MCP -> SQLiteRepository -> SQLite
```

Terraform provisions DSQL, ECR, the AgentCore execution role, Runtime, and
Runtime Endpoint in `us-west-2`. The ARM64 image contains the same project wheel
used by Harbor, co-locates the Billing MCP server, and is deployed by immutable
ECR digest. A manifest binds source revision, wheel digest, AgentConfig
fingerprint, and image digest.

The Runtime uses a custom `billing_runtime` database role with `dsql:DbConnect`;
admin connectivity exists only in operator bootstrap/reset commands. The direct
smoke proves Bedrock, LangGraph, MCP, and DSQL connectivity through a completed
`get_invoice`, not prompt quality.

The target decision is recorded in [ADR 0003](decisions/0003-aws-target-architecture.md)
and Phase 3 details in [ADR 0005](decisions/0005-phase-3-agentcore-dsql.md).

The parity boundary is:

| Concern | Production | Simulation |
| --- | --- | --- |
| Agent code | Same | Same |
| Candidate AgentConfig | Same | Same |
| Prompt and model | Same | Same |
| MCP tool schemas | Same | Same |
| Harness | Same | Same |
| Repository | DSQL | SQLite |
| World state | Live/demo AWS | Frozen Scenario |
| Runtime | AgentCore | Harbor |

Only the world and runtime boundary changes; the candidate configuration being
evaluated does not.

## Local Phase 2 benchmark

```text
Harbor main task container (no network)
  instruction + candidate AgentConfig + uploaded project wheel
                       |
                       v
                LangGraph agent
                       | localhost MCP
                       v
 Billing MCP -> BillingService -> SQLiteRepository -> /app/data/billing.db
                                                        |
                                      Harbor artifact transfer
                                                        v
                      separate verifier container (no network, read-only SQL)
                                                        |
                                                        v
                                      reward.json + diagnostics.json
```

The candidate cannot read the Scenario, expected invariants, verifier source, or
verifier outputs. Its one-shot runner emits response prose, ordered tool calls,
and the candidate fingerprint, but only transferred SQLite state is scored. The
verifier checks the database and invariant cardinalities independently and fails
closed.

Runtime dependencies are exported from `uv.lock` into the main image. The exact
project wheel and Git-revision candidate config are uploaded per run, which keeps
the candidate identity coupled to the evaluated code without allowing task
containers to access external package indexes. A compact build manifest binds
the clean Git revision, candidate fingerprints, and exact wheel SHA-256; the
adapter records the same values in agent logs and Harbor result metadata. The
first benchmark uses a single main container and localhost MCP; Compose remains
a future topology option.

## Failure-to-regression lifecycle

```text
production trace                         future
      |
trace normalizer                         future
      |
normalized Scenario                      Phase 0 contract
      |
candidate Harbor task                    Phase 2 for disputed-refund
      |
Scenario + candidate AgentConfig         Phase 2 for disputed-refund
      |
Harbor trial + hidden state verifier      Phase 2 for disputed-refund
```

`Scenario` is intentionally independent of AgentCore/CloudWatch trace formats
and Harbor task formats. Trace normalization and generic Harbor rendering are
separate, downstream responsibilities. Phase 2 deliberately hand-authors only
the canonical task.

## Scenario and trial identity

```text
Scenario
    = task + deterministic world + provenance

Candidate AgentConfig
    = source + model + prompt + tools + harness being evaluated

Trial
    = Scenario + Candidate AgentConfig
```

The configuration that originally produced a failure may be recorded as optional
scenario provenance. It explains the scenario's origin but does not constrain the
candidate configuration selected for a future trial.

The runtime exports asynchronous, side-effect-free graph factories instead of a
module-level graph or `langgraph.json`. MCP discovery and model configuration are
runtime operations; an import-time graph would require fragile external side
effects.

The Harbor boundary and its accepted Oracle/bad/correct outcomes are recorded in
[ADR 0004](decisions/0004-phase-2-harbor-benchmark.md).

## Phase 4 detection path

```text
AgentCore invocation
  | ADOT + OpenInference LangChain spans
  v
CloudWatch Transaction Search (100% demo indexing)
  |
AgentCore online evaluation (100% demo sampling)
  |
deterministic Lambda evaluator
  |
correlated PASS / FAIL / NOT_APPLICABLE evidence
```

The invocation span carries a session identifier plus bounded `agentops.*`
identity attributes: source revision, AgentConfig fingerprint, model, prompt,
tool version, harness, and billing policy. OpenInference tool spans provide the
only policy inputs. Full AgentConfig JSON, credentials, chain-of-thought, and
unrelated request metadata are not captured.

The evaluator treats a successful refund of any observed disputed invoice as a
failure, requires a successful escalation otherwise, and reports unsupported
telemetry separately from policy failure. The output is correlated to the exact
runtime, candidate, trace, and session before an evidence pointer is written.
That pointer is not a Scenario and does not reconstruct benchmark state.

This separates two jobs:

| Boundary | Purpose | Authority |
| --- | --- | --- |
| Online evaluation | Detect policy evidence in production-shaped traces | Trace evidence |
| Harbor verifier | Prevent regressions using isolated final state | SQLite world state |

Transaction Search indexing, trace content capture, and online sampling are all
set to 100% only for the synthetic demo. Trace normalization into the stable
Scenario contract is deliberately deferred to Phase 5. The decision is recorded
in [ADR 0006](decisions/0006-phase-4-observability-online-evaluation.md).
