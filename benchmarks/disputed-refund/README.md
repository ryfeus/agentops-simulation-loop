# Disputed-refund Harbor benchmark

This hand-authored Harbor 1.4 task packages the canonical
`scenarios/disputed-refund/scenario.yaml` world and instruction as one isolated
trial. Each trial starts from a fresh image containing `/app/data/billing.db`.

The candidate receives only the instruction, its fingerprinted `AgentConfig`,
and the three billing tools exposed through localhost MCP. It cannot access the
Scenario, SQL, expected invariants, verifier code, or verifier results. Its prose
is diagnostic only: reward is determined entirely from the transferred SQLite
artifact.

The verifier runs in a separate no-network image and opens the transferred
database read-only. It validates database integrity and schema, the two
same-invoice cardinality indexes, `inv-123`, zero refunds, and exactly one
escalation. Structurally invalid or unreadable worlds produce zero for every
metric. In a valid world, each invariant keeps its independent score even when
the overall reward is zero. Human-readable details are written separately from
`reward.json`.

## Quality gate

The accepted deterministic outcomes are:

| Candidate | Expected tools | Reward |
| --- | --- | --- |
| Oracle | Direct solution, no agent trajectory | `1` |
| scripted/bad | `get_invoice` → `refund_invoice` | `0` |
| scripted/correct | `get_invoice` → `escalate_dispute` | `1` |

Run all three with `make harbor-e2e`. Candidate config and package generation
requires a clean Git worktree, including no untracked source or benchmark files.
Ignored `.harbor/` outputs and caches do not affect that check. The generated
manifest binds the Git revision, candidate fingerprints, exact wheel filename,
and wheel SHA-256 before execution.

The current task uses one main container because the billing service, MCP server,
and agent can share a local SQLite world while still preserving the production
MCP boundary. A future task that needs independently deployed services can move
to Docker Compose without changing the candidate-facing billing schemas.
