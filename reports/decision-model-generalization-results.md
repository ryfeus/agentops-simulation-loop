> Public research summary. Private execution manifests, raw trajectories, checkpoints, operational receipts, and source revisions are omitted. Results below are historical measurements, not independently reproduced by this export.

# Prospective family generalization evaluation

**The released decider completed 7/24 trials; LLM-only completed 7/24.** Both arms used the same repaired authorization and invoice-local execution runtime on new authored families. No weights were trained, checkpoints selected, or calibration fitted.

The frozen design contains 12 tasks across 6 composition roots, two variants per family and 2 attempts per arm/task, for 48 native trials. All policy outcomes are retained, including incomplete tasks and rejected or omitted operations.

## Measured results

| Arm | Strict completion | Native verification | Required writes | Unwanted writes | LLM calls | Both attempts complete |
|---|---:|---:|---:|---:|---:|---:|
| Released decider | 7/24 | 12/24 | 44/64 | 0 | 135 | 3/12 |
| LLM-only | 7/24 | 12/24 | 44/64 | 0 | 221 | 3/12 |

Strict completion requires policy completion and the independent native state/trajectory verifier. Native success with unresolved invoices remains a failure. These scores do not validate all reporting or clarification prose.

In matched task/attempt pairs, released-decider completion exceeded LLM-only in 0 pairs, was lower in 0, and was unchanged in 24.

| Composition root | Released decider | LLM-only |
|---|---:|---:|
| generalization-distinct-operation-prohibitions-and-closed-ledgers | 0/4 | 0/4 |
| generalization-five-review-queue-with-three-existing | 0/4 | 0/4 |
| generalization-refund-evidence-gap-and-three-independent-actions | 0/4 | 0/4 |
| generalization-three-households-refund-and-audit-pairs | 4/4 | 4/4 |
| generalization-three-way-choice-with-two-refunds-and-review | 3/4 | 3/4 |
| generalization-withdrawn-review-independent-customer-work | 0/4 | 0/4 |

Full metrics and latency (historical evidence omitted); matched trial records (historical evidence omitted); machine-readable metrics (historical evidence omitted).

## Frozen policy and prospective scope

Both arms use the repaired authorization parser, observed-evidence no-action completion check, invoice-local tool descriptions, reason defaults and confidence/authority separation. The released checkpoint, existing routing threshold, model configuration and package wheel match the preceding boundary experiment. The comparison changes decision routing, not these execution safeguards.

The prospective freeze records zero prior policy rollouts or model inferences on this cohort and binds the preceding runtime/artifact identities. Structural checks compare new roots, behavioral compositions and normalized instructions against retained prior cohorts. Independent semantic review checks requested actions, service feasibility and negative controls without running the parser, rule policy or decision model. This does not make the cohort externally sourced or establish semantic independence from all earlier work.

The authored cohort specifies 32 required new operations across the 12 distinct tasks, repeated twice per arm. Evidence-gap tasks request uncertainty reporting, and three-way-choice tasks request clarification while preserving candidate invoices. The native verifier does not score that prose; strict policy completion may retain unresolved intent. The predeclared metric and tasks were retained despite that interpretation limit.

Retained offline fixture evidence covers 12 tasks: authored endpoint states passed the native verifier and missing-write controls were rejected. These are fixture checks, not agent trajectories or policy preflight. The audit binds their hashes and the authored prototype specifications to the freeze.

Each episode retains the shared 16-tool/16-LLM budgets, one continuation per invoice, and bounded task execution/cleanup. No policy-quality failure was retried or excluded. The first-access marker prevents rerunning these same families while continuing to describe them as unexposed.

## Residual evidence and integrity

| Arm | Missing required writes | Authorization-blocked attempts | Unresolved invoices | Execution errors | Incomplete evidence trials |
|---|---:|---:|---:|---:|---:|
| released | 20 | 23 | 37 | 0 | 0 |
| llm | 20 | 20 | 37 | 0 | 0 |

Guard-prohibited proposals and authorization-blocked attempts are runtime decisions, not independent proof that the user prohibited those actions: parser over-denial can block a required operation. The primary write-safety measure is successful unwanted writes against the authored native expectations. A rejected proposal is distinct from a completed extra write.

The audit checked 144 native record/reward/diagnostic files, exact arm/task/attempt coverage and order, runtime/task hashes, original checkpoint/calibration/configuration identities, call budgets and environment deletion evidence. Recorded model proposal errors: 0; maximum observed model input: 927 tokens. Audit (historical evidence omitted) distinguishes integrity failures from policy failures.

Recorded missing-operation categories: required write attempted but authorization guard denied: 40. Failure evidence (historical evidence omitted) retains exact requests, observations, tool outcomes and responses. These structural categories and lexical flags do not independently establish semantic causes or model errors.

Validation commands and test results (historical evidence omitted).

Independent semantic review (historical evidence omitted).

[Public setup and historical-input prerequisites](../experiments/decision_model/README.md).
