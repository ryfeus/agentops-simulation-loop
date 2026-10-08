# Phase 0 contracts

All models reject unknown fields. The strict boundary makes accidental schema
drift visible before a scenario reaches a runtime or benchmark system.

## Billing MCP surface

The public agent-visible tool surface is exactly:

```text
get_invoice(invoice_id)
refund_invoice(invoice_id, reason)
escalate_dispute(invoice_id, reason)
```

Requests and results are typed in `agentops_demo.contracts.billing`. Amounts are
parsed as `Decimal` and emitted as strings, and currency codes are three uppercase
ASCII letters. The business invariant is:

> A disputed invoice must not be automatically refunded. It must be escalated
> for specialist review.

Phase 0 documents this rule and encodes expected scenario outcomes. Phase 1's
`BillingService` makes the rule an explicit experiment variable: permissive mode
reproduces the failure, while enforced mode blocks it.

## Repository boundary

`BillingRepository` is an asynchronous protocol containing only `get_invoice`,
`refund_invoice`, and `escalate_dispute`. Phase 3 implements it with Aurora DSQL
for production while simulation continues using seeded SQLite. Neither storage
choice changes agent-visible schemas.

The Phase 1 SQLite implementation creates refund and escalation records with
stable local IDs, updates invoice status transactionally, and reports missing or
duplicate operations through coded domain errors. Unique database indexes also
enforce at most one refund and one escalation per invoice when application
guards are bypassed. These persistence and error details are not added to the
agent-visible schema.

The DSQL implementation uses IAM-authenticated, short-lived connections, named
uniqueness constraints, exact numeric amounts, and bounded OCC retry. Its
`billing_runtime` database role is mapped only to the AgentCore execution role;
administrator connectivity is restricted to deployment bootstrap and reset
commands.

## Agent configuration identity

`AgentConfig` requires source revision, model provider and identifier, prompt
version, tool version, and harness framework/version. Its `fingerprint()` is a
lowercase SHA-256 digest of canonical sorted JSON for those fields. Credentials,
environment settings, and secrets are neither accepted nor fingerprinted.

Phase 1 graph factories require the complete candidate configuration. Runtime
construction resolves the prompt from it and rejects unsupported tool or harness
versions rather than treating the fingerprint as passive metadata. Injected
models keep deterministic tests cheap; the Bedrock construction path remains
`create_model(config)`.

Phase 2 also permits exactly `scripted/bad` and `scripted/correct` through that
same model factory. These are deterministic candidate identities used to verify
the benchmark pipeline; all other scripted model IDs and non-Bedrock providers
are rejected.

## Normalized Scenario

A Scenario contains:

- schema version and filesystem-safe identifier;
- the user-visible instruction;
- provider-neutral source and optional evaluation provenance;
- the originating agent configuration, when known;
- deterministic initial billing state;
- the observed failing tool trajectory; and
- deterministic expected invariants.

Schema version `1` supports the three billing tool calls and primitive billing
invariants: `no_refund`, `must_refund`, `no_escalation`, `must_escalate`,
`invoice_status`, and `unchanged`. `must_*` invariants require a single new
record relative to the initial world; `no_*` invariants require no new record;
`unchanged` compares the invoice row and all associated mutation records.
Entity and invoice references are validated across the complete document.
Arbitrary verifier code is not allowed inside YAML.

Static benchmark scenarios may omit `observed_failure` and instead include
typed `benchmark` metadata for category, archetype, difficulty, allowed
mutation targets, and trajectory expectations. Trace-derived Scenario sources
must retain observed failure evidence. This prevents a static task from
fabricating production failure provenance while retaining a single Scenario
intermediate representation.

`ScenarioSource.kind` is either `synthetic` or `trace`. Synthetic sources cannot
carry a trace reference; trace sources require a nonblank, provider-neutral
reference. Either source may contain optional `EvaluationEvidence` with a
nonblank evaluator, score from zero through one, and optional reason.

Entity IDs occupy separate namespaces. IDs must be unique within customers,
invoices, refunds, or escalations, but the same value may appear in different
entity types. References remain strict: invoices identify existing customers,
and refunds and escalations identify existing invoices.

Invariant cardinality is exact:

```text
no_refund(invoice_id)     -> final refund count for invoice == 0
must_escalate(invoice_id) -> final escalation count for invoice == 1
```

An optional `originating_agent_config` records the configuration associated with
the scenario's origin. It is provenance, not the candidate configuration for a
trial. A future trial combines a reusable Scenario with a separately selected
candidate `AgentConfig`.

The instruction and billing request arguments are agent-visible. Source metadata,
the originating configuration fingerprint, the observed trajectory, initial-state
seed data, and expected invariants belong to normalization, evaluation, or
verification infrastructure. Harbor-specific fields and AgentCore trace payloads
are excluded.

The Phase 2 Harbor adapter preserves this split: a candidate receives the
instruction and its separate `AgentConfig`, while the initial state is baked into
the task world and expected invariants remain inside the separate verifier.

Phase 0 intentionally records observed tool names and arguments plus deterministic
initial state, but does not normalize tool results or state changes. Their future
representation will be derived from real AgentCore/LangGraph trace shapes during
trace conversion instead of being guessed in advance.
