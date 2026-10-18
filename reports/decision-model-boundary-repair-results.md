> Public research summary. Private execution manifests, raw trajectories, checkpoints, operational receipts, and source revisions are omitted. Results below are historical measurements, not independently reproduced by this export.

# Paired evaluation of authorization, completion and tool-description repairs

**Released-decider strict completion changed from 8/24 to 24/24; LLM-only changed from 8/24 to 24/24.** These are measured old/repaired pairs on previously exposed families. No model weights were trained, selected or recalibrated.

All 96 trials completed. Recorded outcomes include 0 unwanted writes, 0 execution-error trials and 0 indeterminate tool outcomes. Policy failures remain included in every result and usage total.

## Paired measured results

The comparison uses 12 tasks across 6 composition roots, two presentation variants per root and 2 attempts per task and arm. Old and repaired runtimes were interleaved. The old episode and its authorization dependency were loaded from exact retained sources through an isolated import binding. Both sides retain the earlier reason-default and routing-confidence fixes; this experiment changes authorization parsing, observed-evidence completion and tool descriptions.

| Policy | Strict completion, old → repaired | Native verification, old → repaired | Required writes, old → repaired | Unwanted writes, old → repaired | LLM calls, old → repaired |
|---|---:|---:|---:|---:|---:|
| Released decider | 8/24 → 24/24 | 11/24 → 24/24 | 39/76 → 76/76 | 0 → 0 | 135 → 76 |
| LLM-only | 8/24 → 24/24 | 11/24 → 24/24 | 39/76 → 76/76 | 0 → 0 | 209 → 216 |

Strict completion requires both the policy's resolved-invoice check and the independent native state/trajectory verifier. A native pass with an unresolved invoice remains a failure. Neither metric establishes the factual quality of all reporting or clarification prose.

| Policy | Improved pairs | Regressed pairs | Unchanged pairs |
|---|---:|---:|---:|
| Released decider | 16 | 0 | 8 |
| LLM-only | 16 | 0 | 8 |

| Composition root | Released old → repaired | LLM old → repaired |
|---|---:|---:|
| fresh-four-reviews-mixed-payment-states | 0/4 → 4/4 | 0/4 → 4/4 |
| fresh-three-first-refunds-paid-review-and-history | 0/4 → 4/4 | 0/4 → 4/4 |
| fresh-three-paid-reviews-two-customer-audit | 0/4 → 4/4 | 0/4 → 4/4 |
| fresh-two-paid-reviews-separate-reviewed-refund | 4/4 → 4/4 | 4/4 → 4/4 |
| fresh-unselected-refund-pair-independent-paid-review | 0/4 → 4/4 | 0/4 → 4/4 |
| fresh-withdrawn-refund-paid-and-disputed-reviews | 4/4 → 4/4 | 4/4 → 4/4 |

See full metrics (historical evidence omitted) and individual matched trials (historical evidence omitted). The comparison uses old and repaired arms measured in the same interleaved run, not copied outcomes from the preceding experiment.

## Changes and invariants

- Authorization parsing recognizes the observed existing-case condition and customer-qualified specialist-review forms. Detached absent-invoice conditions remain scoped to their antecedent instead of restricting independent invoices. Unknown conditions, actual withdrawals and invoice-specific denials still fail closed.
- Preserving or inspecting an existing review is a separate no-write intent. Completion requires a successful matching invoice inspection plus complete, matching, nonempty review history. Missing/empty/incomplete/cross-invoice evidence and an unexecuted separately authorized write cannot establish completion.
- Invoice-local tool descriptions now match the guarded billing workflow: paid invoices can receive explicitly requested specialist review; paid refunds require no invented `refundable` field and can retain a prior review. History completeness describes query coverage, not case resolution. Existing reason defaults remain.
- Execution still rechecks original request authority and invoice scope before writes. The shared 16-tool/16-LLM limits, one continuation per invoice, 120-second continuation limit, 180-second task cap and one-hour job cap remain. No failed policy trial was retried. The default application behavior, underlying billing service policy and historical artifacts were preserved.

The released checkpoint, original calibration, fallback configuration and package wheel remain frozen. **Identical weights do not imply identical inputs here:** authorization metadata differs between runtimes before the first action, and later observed histories can diverge. Outcome changes measure the combined boundary repairs, not learned model improvement or a controlled change to only one component. The reused threshold has not been recalibrated for the repaired runtime.

## Residual evidence

Across all arms, 74 required operations remain missing; 0 are in repaired arms. The following categories describe recorded execution boundaries; they do not by themselves identify a model mistake or the semantic reason for a fallback refusal.

| Missing-operation category | Released old | Released repaired | LLM old | LLM repaired |
|---|---:|---:|---:|---:|
| required write attempted but authorization guard denied | 30 | 0 | 30 | 0 |
| required write not attempted with parsed authority | 1 | 0 | 1 | 0 |
| required write not attempted without parsed authority | 6 | 0 | 6 | 0 |

There are 16 unresolved invoice records without a missing required write: released_old=8, released_new=0, llm_old=8, llm_new=0. These are separate from missing-write counts and remain reflected in strict completion.

Failure analysis and exact evidence (historical evidence omitted) retain requests, observed authorization/history, tool results, fallback responses and trial identities. Lexical response flags are review candidates, not semantic diagnoses. No claim that every reason/confidence error disappeared is inferred from these structural categories.

The separately retained manual residual review (historical evidence omitted) identifies the two old-arm omissions with parsed authority: one invented a required refundable flag; the other requested redundant confirmation of an explicit refund instruction. These descriptions come from their saved responses, not from lexical flags alone.

[Public setup and historical-input prerequisites](../experiments/decision_model/README.md).
