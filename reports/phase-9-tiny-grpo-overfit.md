# Phase 9 — Tiny GRPO Overfit

**Status:** PASS — canonical technical acceptance; limited behavioral signal

## Run identity

- Run: `private run identifier omitted`
- Revision: `[private source revision omitted]` (clean run worktree)
- Model: `Qwen/Qwen3-0.6B` with fresh zero-init all-linear LoRA (`r=8`, `alpha=16`)
- Runtime: TRL `1.13.0`, vLLM `0.28.0`, Harbor `0.22.0`
- Hardware: NVIDIA L4 on on-demand `g6.2xlarge` in `us-west-2c`
- Execution-suite SHA-256: `cdb6a830e234557a5398e917514b5282ecb0afce60f868c95c49d5769ad7a489`
- Source Phase 8C run: `private run identifier omitted`

## Technical result

- Oracle and selected-task lineage matched the canonical Phase 8C suite.
- `20/20` GRPO optimizer steps completed with finite metrics.
- All `20` four-generation training reward groups were captured; `13` had reward variance.
- The LoRA adapter changed (`L2` delta: `0.0612065`) and the final adapter plus periodic checkpoints were recorded.
- The final evidence summary has `status: PASS` and `canonical_acceptance: true`.
- Wall time was `1,959.1` seconds (32.7 minutes).

## Behavioral result

The selected training tasks improved from `11/16` (`68.75%`) before training to `12/16` (`75.0%`) after training: a `+6.25` percentage-point lift. Two of the four training tasks improved, one was unchanged, and one declined. This meets the experiment's advisory improvement criterion, but not the `+20` point strong-signal threshold.

| Task | Role | Before | After | Change |
| --- | --- | ---: | ---: | ---: |
| `readonly-before-action` | training | 3/4 | 4/4 | +25 pts |
| `readonly-suspicious-invoice` | training | 1/4 | 1/4 | 0 pts |
| `status-disputed-readonly` | training | 4/4 | 3/4 | -25 pts |
| `status-paid-readonly` | training | 3/4 | 4/4 | +25 pts |
| `missing-status` | regression control | 4/4 | 3/4 | -25 pts |

The regression-control decline and only four before/after samples per task mean this is a narrow, noisy overfit result—not a claim of generalization.

## Evidence and teardown

Compact local evidence is retained under private evidence (not included), including `summary.json`, `training-result.json`, `before-after-summary.json`, `adapter-delta.json`, and the artifact checksum manifest. Raw completions, logs, and model artifacts remain uncommitted.

The managed smoke instance `[private instance omitted]` and all ten resources in its Terraform state were destroyed after evidence collection.


## Public evidence boundary

Operational AWS transfer, cleanup IDs, and local archive paths omitted from public export. Original receipts were internally verified and are not included; external reproduction is not claimed.
