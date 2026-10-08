# Phase 8B — TRL / Harbor billing smoke

Phase 8B is a one-step integration smoke, not a billing-training experiment.
TRL owns Qwen3-0.6B's tool tokens, rollout, log probabilities, and GRPO update.
Harbor owns the existing `paid-refund-direct` Docker sandbox, SQLite state, and
verifier reward. The sandbox has no network and receives no cloud credentials.

The only model-visible tools are `get_invoice`, `refund_invoice`, and
`escalate_dispute`. Each invokes the in-sandbox bridge, which uses the existing
`BillingService` and `SQLiteBillingRepository` in permissive mode and records
logical calls in `/app/data/agent-run.json` for the existing verifier.

Run `make rl-harbor-config-check` before AWS work. The live controller requires
a clean committed worktree because it builds and records a provenance-bound
project wheel. Evidence is written under `.rl-smoke/phase8b/runs/<run-id>/`.

TRL 1.13 constructs Harbor environments directly, bypassing the normal Harbor
Trial lifecycle. Harbor 0.22 Docker expects Trial log mounts and does not apply
the task's separate-verifier topology through that path. Phase 8B restores the
log mounts and derives a temporary shared-verifier execution copy while keeping
the canonical benchmark unchanged. Remove this compatibility layer once
upstream supports both Trial mounts and verifier topology in TRL's HarborEnv.
