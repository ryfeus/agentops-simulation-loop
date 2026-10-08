# Phase 9 Hardening — Persisted-Adapter Reproduction

**Status:** Canonical technical PASS; adapter retained and verified locally; G6 destroyed.

## Why this run was needed

The historical `private run identifier omitted` retained only an artifact manifest, not the
adapter or raw semantic rollout files. A later attempt, `private run identifier omitted`, also
remained incomplete after its instance was terminated. Neither is a Phase 9b source.
The new run kept the same Phase 9 task selection, Qwen3-0.6B, zero-init all-linear LoRA
(`r=8`, `alpha=16`), seed, learning rate, 20-step limit, Harbor/verifier path, and
colocated vLLM settings.

## Verified run

- Run: `private run identifier omitted`
- Clean source revision: (private revision omitted)
- Runtime: TRL `1.13.0`, vLLM `0.28.0`, Harbor `0.22.0`; on-demand L4 `g6.2xlarge` in `us-west-2c`
- Execution-suite SHA-256: `cdb6a830e234557a5398e917514b5282ecb0afce60f868c95c49d5769ad7a489`
- Exact training-task-set SHA-256: `3c30b473b7e3c880ea036355b37c952756dc2cd8bca0473f598f2dfa54eb5016`
- 20 optimizer steps, 20 reward groups, 80 training rollouts, and 20 rollouts each before and after training
- Seven final-adapter files retained under the ignored local path
  private evidence (not included); all sizes and SHA-256
  digests match `artifacts-manifest.json`.

The local summary reports `status: PASS`, `canonical_acceptance: true`,
`adapter_persisted_locally: true`, and `adapter_checksums_verified: true`. The independent
Phase 9 source check and Phase 9b no-rollout configuration check pass. The managed
Terraform stack was destroyed (10 resources), and EC2 reports the instance terminated.

## Behavioral reading

The four training tasks passed 13/16 before and 13/16 after (81.25% both times); none
improved in this small paired sample. Eight training reward groups had variance, and the
LoRA weights changed (L2 delta `0.05982266`). The selected `missing-status` control moved
from 4/4 to 3/4, which is not the defined catastrophic-regression threshold. This run
establishes a technically valid, persisted adapter—not a supported behavioral lift or
generalization claim. The subsequent 160-rollout comparison is documented in
[Phase 9b — Frozen Baseline vs. Persisted Phase 9 Adapter](phase-9b-paired-evaluation.md).

## Retention issue fixed

The remote training completed, but the first local controller lost its exported AWS
credentials while polling. A resumable SSM fetch then downloaded the adapter. Its initial
source check exposed an evidence bug: redaction reformatted the safe task-set JSON,
changing its byte-exact lineage hash. The controller now preserves safe JSON bytes,
re-fetches an invalid task-set copy against the original SHA, and derives summary
cleanliness from the immutable run identity rather than the worktree at fetch time.
The Phase 9b adapter check now distinguishes `adapter_model.safetensors` from the
accompanying `training_args.bin`.

Raw rollouts, logs, GPU traces, and adapter files remain local and unversioned under
`.rl-smoke/`; this report contains only compact, sanitized results.


## Public evidence boundary

Operational AWS transfer, cleanup IDs, and local archive paths omitted from public export. Original receipts were internally verified and are not included; external reproduction is not claimed.
