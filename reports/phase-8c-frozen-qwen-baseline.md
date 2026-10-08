# Phase 8C — Frozen Qwen Billing Baseline

**Status:** PASS — canonical acceptance

## Run identity

- Revision: `[private source revision omitted]` (clean worktree)
- Model: `Qwen/Qwen3-0.6B` with zero-init LoRA (`r=8`, `alpha=16`, `all-linear`)
- Runtime: TRL `1.13.0`, vLLM `0.28.0`, Harbor `0.22.0`
- Hardware: NVIDIA L4 on `g6.2xlarge` in `us-west-2a`
- Execution-suite SHA-256: `cdb6a830e234557a5398e917514b5282ecb0afce60f868c95c49d5769ad7a489`
- Base-suite SHA-256: `4ac6da802a651bbf30513b2f5dd04eb4c489e7a8a6750bc008eceb4dc5803f9e`

## Acceptance results

- Oracle gate: **25/25** derived shared-verifier tasks passed.
- Frozen evaluation: **100/100** logical rollouts completed (25 tasks × 4 attempts).
- Infrastructure-invalid rollouts / retries: **0 / 0**.
- Frozen-policy invariants: no `train()` call, global step `0`, no optimizer, no checkpoint.
- Wall time: **1,633.3 seconds** (27.2 minutes); **0.0612 rollouts/second**.
- Canonical acceptance: **true**.

The baseline’s policy score is intentionally separate from harness acceptance: overall reward pass rate was **16%**. This is a valid frozen baseline, not a trained policy result.

## Outcome distribution and Phase 9 signal

- Always-fail tasks: 20
- Mixed-reward tasks: 4
- Always-pass regression controls: 1
- `phase9_ready`: **false** — the four mixed tasks all belong to the `read-only` archetype, rather than the roughly five behaviorally diverse tasks needed for the proposed Phase 9 pool.

The mixed tasks were `readonly-before-action`, `readonly-suspicious-invoice`, `status-disputed-readonly`, and `status-paid-readonly`; each passed 3/4 attempts. The `missing-status` task was the stable always-pass regression control.

## Initial attempt and correction

The first G6 attempt loaded the isolated runtime and Qwen weights but stopped before rollouts because TRL 1.13 requires a positive `GRPOConfig.max_steps` when prompts come from an environment reset. Commit `856e51f` added the fixed 25-task evaluation horizon and a unit assertion. The retry passed 320 CPU unit tests, the locked configuration check, the Oracle gate, and canonical G6 evaluation.

## Evidence

- Sanitized canonical summary (private evidence omitted)
- Sanitized task aggregate (private evidence omitted)
- Sanitized Phase 9 candidates (private evidence omitted)
- Canonical baseline configuration (private evidence omitted)

The disposable G6 instance was terminated and Terraform state was empty after evidence collection. Full raw evidence remains local under private evidence (not included) and is intentionally not committed.


## Public evidence boundary

Operational AWS transfer, cleanup IDs, and local archive paths omitted from public export. Original receipts were internally verified and are not included; external reproduction is not claimed.
