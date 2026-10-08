# ADR 0007: Trace evidence to Harbor regression candidates

- Status: Accepted
- Date: 2026-09-10

## Decision

Phase 5 captures the entire tiny synthetic billing world before each taskifiable
AgentCore invocation and records it only as hidden trace evidence. Taskification
requires a correlated online `FAIL` result plus that exact source trace; it never
queries current DSQL state to reconstruct history. The snapshot is canonical JSON,
SHA-256 bound, schema-versioned, and limited to 32 KiB.

`DisputePolicyCompliance` is converted deterministically into the existing Phase
0 `Scenario` contract. No model generates scenario semantics, verifier logic, or
missing tool arguments. Scenario rendering is billing-v1-specific and preserves a
separate, no-network final-SQLite verifier.

Rendering does not validate a candidate. `validate-scenario` verifies the
canonical contract only; `reproduce` and `validate-benchmark` execute a generated
invariant-driven Oracle, a scripted/correct agent, and the supported
scripted/bad-origin calibration. The generated Oracle supports only `no_refund`
and `must_escalate`, creating deterministic per-invoice escalation IDs and
rejecting unsatisfiable captured states.

Generated tasks remain ignored candidates until all three gates succeed. Their
`reproduction.json` is `VALIDATED` only when Oracle and known-good reward `1` and
the scripted/bad calibration rewards `0`; synthetic fixtures are explicitly
`calibration`, not `exact_source` replay. Production data requires an explicit
sanitization policy before taskification. Bedrock candidates are `UNSUPPORTED` in
Harbor: the benchmark isolation boundary remains no-network.

The report contract makes those claims structural: failed trial observations are
retained before expectations are asserted, source-replay counters exclude Oracle
and controls, and `VALIDATED` cannot be serialized without all required results.
Calibration records a clean execution package's Git revision and SHA-256
separately from its source candidate revision. Bedrock is classified unsupported
before any Harbor controls run.

## Consequences

Online evaluation is the detector and Harbor is the regression test. Automatic
promotion, worker scaling, UI, semantic deduplication, and networked benchmark
replay remain out of scope.
