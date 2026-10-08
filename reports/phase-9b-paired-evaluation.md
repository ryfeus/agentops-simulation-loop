# Phase 9b — Frozen Baseline vs. Persisted Phase 9 Adapter

**Status:** Technical PASS; advisory behavioral-improvement signal; human review required. The disposable G6 stack was destroyed.

## Run and provenance

- Run: `private run identifier omitted` on clean revision `[private source revision omitted]`
- Source adapter: verified Phase 9 run `private run identifier omitted`; Qwen3-0.6B with its persisted final LoRA adapter, not a fresh or stacked adapter
- Execution-suite SHA-256: `cdb6a830e234557a5398e917514b5282ecb0afce60f868c95c49d5769ad7a489`
- Source task-set SHA-256: `3c30b473b7e3c880ea036355b37c952756dc2cd8bca0473f598f2dfa54eb5016`
- Adapter-config SHA-256: `8ad92883f0fa9b5dc5e4f7bf21fb7ddec3fcd5dee87b2584000b5646145749e5`
- Adapter-weights SHA-256: `5bf415674abd6a9b5f7f1ccd5cc7f7941c9b0e7554dce7de2f2039ecc40b6c2d`
- On-demand L4 `g6.2xlarge` in `us-west-2c`; four native TRL evaluation passes per policy, four generations per task per pass, seeds `20260923`–`20260926`

The fresh zero-init LoRA baseline completed before the trained policy began. Each policy produced exactly 80 valid rollouts: 16 per task across the four selected training tasks and the disjoint `missing-status` control. Both policy results report `training_performed: false`, global step zero, no optimizer, and no checkpoint. The trained policy loaded the source PEFT adapter, recorded native colocated-vLLM synchronization, and passed adapter hash checks before and after each pass.

## Measured outcome

| Task | Baseline | Trained | Change |
| --- | ---: | ---: | ---: |
| `readonly-before-action` | 10/16 | 13/16 | +3 |
| `readonly-suspicious-invoice` | 12/16 | 11/16 | −1 |
| `status-disputed-readonly` | 16/16 | 16/16 | 0 |
| `status-paid-readonly` | 15/16 | 16/16 | +1 |
| **Four training tasks** | **53/64 (82.8%)** | **56/64 (87.5%)** | **+3/64 (+4.7 points)** |
| `missing-status` control | 15/16 | 15/16 | 0 |

Across all five tasks, the micro pass rate was 68/80 (85.0%) for baseline and 71/80 (88.75%) for trained. The comparison marks `behavioral_improvement_supported: true`: aggregate training rate rose, two training tasks improved, no training task fell by 25 points, and the selected control did not regress catastrophically. This is a small, noisy evaluation—not a claim of statistical significance or generalization. One training task declined by 1/16, and the control held steady; candidate selection still requires human review.

## Runtime fix and validation

An earlier attempt, `private run identifier omitted`, stopped after 40 baseline rollouts because successive colocated-vLLM trainers retained GPU memory in one Python process. Revision `98dddbe` runs each native `GRPOTrainer.evaluate()` pass in a separate short-lived process, preserving the fixed seeds and evaluation configuration while releasing CUDA memory between passes. The retry completed all eight passes. Local lint, 345 CPU unit tests, and the locked Phase 9b source/config check passed before the retry.

The controller-fetched summary reports `status: PASS`, 80/80 policy coverage, a clean worktree, and `human_review_required: true`. Its `canonical_acceptance` field remains `false` by Phase 9b controller design; Phase 9b acceptance is represented by the validated `status: PASS` rather than that Phase 8C-style flag. Terraform destroyed all 10 managed resources, its state is empty, and EC2 reported the G6 instance `terminated`.

Raw rollouts, logs, GPU traces, and the local summary are retained only under the ignored path private evidence (not included); they are not versioned with this report.


## Public evidence boundary

Operational AWS transfer, cleanup IDs, and local archive paths omitted from public export. Original receipts were internally verified and are not included; external reproduction is not claimed.
