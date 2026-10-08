# ADR 0004: First local Harbor benchmark

- Status: Accepted
- Date: 2026-09-07

## Context

Phase 2 must turn the canonical disputed-refund Scenario into a reproducible
quality gate without letting the candidate inspect expected outcomes. Candidate
execution should reuse the Phase 1 LangGraph and MCP path, and the benchmark must
prove that a solvable correct state passes while the known failure does not.

## Decisions

1. The canonical Scenario is packaged as one hand-authored Harbor task using
   schema 1.4 and Harbor `>=0.22,<0.23`.
2. The main task image contains dependencies exported reproducibly from
   `uv.lock`; Harbor uploads the exact local project wheel and installs it with
   `--no-deps`.
3. Candidate configs are generated from a clean Git worktree with tracked and
   non-plan untracked changes rejected. Untracked Markdown plan documents do not
   alter executable provenance. Both candidates use `billing-v1`; only the
   scripted model ID differs.
4. `LangGraphBillingAgent` validates the wheel, complete configuration,
   fingerprintable runtime identity, and Harbor `provider/model_id` before
   setup.
5. Candidate execution receives only its config, the instruction, and localhost
   MCP. SQL, shell, filesystem tools, Scenario provenance, invariants, and
   verifier data are not agent inputs.
6. `/app/data/billing.db` is the sole scored artifact. Harbor transfers it at
   the same absolute path to a separate no-network verifier image.
7. The verifier uses only the Python standard library and opens SQLite read-only.
   It fails closed with all numeric metrics zero on structurally invalid or
   unverifiable state, while preserving independent invariant scores for a valid
   world whose expected cardinalities are not all satisfied.
8. Reward is based on final state, never final-response prose. Ordered tool calls
   and configuration fingerprints are retained only as diagnostics.
9. The Oracle mutates SQLite directly. It proves task solvability and verifier
   correctness, not production-path parity.
10. The initial topology is one main container containing the candidate,
    localhost MCP server, service, and SQLite repository. Docker Compose is
    deferred until a benchmark requires distinct service containers.
11. A generated manifest binds the Git revision, candidate fingerprints, wheel
    filename, and streaming SHA-256. The adapter independently hashes the wheel
    before upload and records the matching provenance in Harbor metadata and
    `/logs/agent/provenance.json`.

## Accepted outcomes

The Docker acceptance gate produced these deterministic results:

| Run | Final trajectory | Reward |
| --- | --- | --- |
| Oracle | direct solution | `1` |
| scripted bad | `get_invoice` → `refund_invoice` | `0` |
| scripted correct | `get_invoice` → `escalate_dispute` | `1` |

## Consequences

Phase 2 provides one repeatable local benchmark and a custom installed-agent
adapter, not a generic Scenario renderer or control plane. Real-model reward
gates, Compose tasks, AgentCore, DSQL, Terraform, UI, observability, trace
conversion, and EC2 workers remain future work.
