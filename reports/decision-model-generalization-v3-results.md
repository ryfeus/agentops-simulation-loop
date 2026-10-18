> Public research summary. Private execution manifests, raw trajectories, checkpoints, operational receipts, and source revisions are omitted. Results below are historical measurements, not independently reproduced by this export.

# Repaired runtime: regression, fresh families, and failure attribution

The repaired runtime passes the entire exposed regression cohort, but the released decider does not improve completion over LLM-only on the new families. Four shared authorization-parser defects block required operations in both arms. Weights and calibration were unchanged; no training took place.

| Comparison | Released decider | LLM-only | Required writes, each arm | Unwanted writes |
|---|---:|---:|---:|---:|
| Exposed regression, 24 tasks × 2 attempts | 48/48 | 48/48 | 132/132 | 0 |
| Fresh complete first pass, 12 tasks | 4/12 | 4/12 | 18/30 | 0 |
| Fresh complete pairs, including available repeats | 7/22 | 7/22 | 33/55 | 0 |

Strict completion and native-verifier pass counts coincide in these comparisons. All 48 regression pairs and all 22 complete fresh pairs tie. The first fresh pass covers both variants of all six families: two families pass in both arms, four fail in both arms. Repetitions and variants are correlated, so these are not 24 independent fresh families.

The fresh run retained **45 of 48 planned trials**. Its final, unmatched LLM-only trial timed out after both required refunds succeeded, with remaining inspection/completion work unfinished. Three planned trials were not started. The runner then stopped at its fixed time bound; no trials were retried or substituted. All retained fresh results are 7/22 for released and 7/23 for LLM-only, with 33/55 and 35/57 required writes respectively. Those unequal denominators must not be treated as a paired performance difference. The complete first pass and complete pairs above avoid that problem.

## What improved on previously exposed tasks

The strict regression gate passed: 96/96 completed episodes, all 264 required writes, zero unwanted writes, and complete native evidence. Twelve duplicate-review proposals rejected by the authorization guard were recovered through the bounded invoice-local fallback. Two additional wrong refund proposals were recovered through low-confidence fallback.

Matched historical subsets improved from 10/12 to 12/12 for the released decider on v1 tasks, while LLM-only stayed 12/12. On the v2 tasks, both arms improved from 20/24 to 24/24. Historical sources and task identities were audited. These are exposed engineering comparisons with provider variation and runtime changes, not randomized evidence of learned improvement.

## Why fresh families still fail

Every one of the 44 missing required writes across retained fresh trials maps to one of these shared parser defects. The fallback requests the correct operation, but the guard rejects it based on the parser's authorization result.

| Defect | Observed consequence | Required repair |
|---|---|---|
| Positive grant followed by an opposite-operation prohibition stays in one clause | “Escalate A, but do not refund A” loses the escalation grant | Parse both operation-specific clauses while preserving the prohibition |
| “Those four invoices” is not resolved to its explicit preceding list | A refund prohibition spreads to an independently authorized invoice | Resolve the unambiguous antecedent and keep its scope local |
| Withdrawal of one operation revokes both operations | Withdrawing review also removes a retained refund grant, or vice versa | Apply withdrawal to the named operation and target only |
| Comma-separated customer-qualified commands are not segmented | The first two requested refunds disappear; the final “and refund” survives | Recognize repeated command boundaries without splitting invoice lists incorrectly |

No observed required-write failure was caused by a fallback call escaping its target invoice. Existing review preservation and pending-approval handling work in the passing families. These results do not justify weakening fail-closed handling of genuinely unresolved conditions or scopes.

The decider still makes mistakes: it proposed eight duplicate reviews on observed existing reviews in the fresh run (four distinct situations repeated). Confidence was approximately 0.605–0.639, below the unchanged 0.7 threshold, and fallback resolved every one. Those are separate model training candidates; the independent refund denied by the parser caused those tasks to fail. The regression had 14 recovered raw model errors: 12 duplicate reviews and two refunds on disputed invoices.

Two Bedrock `ServiceUnavailableException` events occurred during released-arm repeats. One followed a successful refund; the successful write remains counted despite an unresolved continuation. The other interrupted completion for an absent invoice. Both coexist with parser failures already present on those tasks. The final unmatched LLM-only trial hit the shrinking task deadline. These provider and budget failures are recorded separately from model and parser errors. LLM prose speculating about account permissions is not evidence of an AWS authorization problem.

See the manual diagnosis (historical evidence omitted) and its JSON for the request, observations, parsed authorization, rejected call, native evidence and resolution for each residual failure.

## Calls, latency, and resources

The released decider used 202 versus 445 LLM calls in regression, a 54.6% reduction. On the complete fresh first pass it used 99 versus 116, a 14.7% reduction, with no completion gain. Across the 22 complete fresh pairs, it used 185 versus 213 calls. Fewer calls do not automatically mean lower end-to-end latency or a measured dollar saving.

| Run/arm, all retained records | LLM calls | Input tokens | Output tokens | Policy p50 / p95 |
|---|---:|---:|---:|---:|
| Regression released, 48 trials | 202 | 373,900 | 25,370 | 15.87 / 20.26 s |
| Regression LLM-only, 48 trials | 445 | 842,302 | 50,259 | 19.11 / 24.34 s |
| Fresh released, 22 trials | 185 | 349,072 | 22,220 | 25.94 / 29.02 s |
| Fresh LLM-only, 23 trials, including timeout | 222 | 423,198 | 26,514 | 21.29 / 23.68 s |

Execution occupied **3,542.12 seconds (59.04 minutes)** across the two paid runs: 2,234.62 seconds regression and 1,307.50 seconds fresh. The shared bound was 3,600 seconds; the frozen fresh allowance was 1,365 seconds with verification/cleanup reserve. Preparation and offline reporting are not included in paid execution wall time. The complete 144-trial design did not fit the allowance; future scheduling should account for these measured durations.

Execution used local MPS and Bedrock `us.anthropic.claude-sonnet-4-6`, with refreshable AWS `default` credentials in a private research account, region `us-west-2`. Total returned usage: **1,054 LLM calls, 1,988,472 input tokens, 124,363 output tokens**. Failed requests can omit provider usage, so these are observed usage totals, not an AWS bill. No EC2/training worker was created; training steps and calibration fits are both zero.

All 141 created trial environments have owned-deletion records. The final Docker inventory is identical to the initial inventory: four preexisting containers preserved, no additional containers. No unwanted writes, indeterminate writes, incomplete tool evidence, or successful scope-bypass calls were observed in retained records.

## Evidence and limitations

The regression audit passes. The fresh full-run audit deliberately **fails completeness**, retaining all nine coverage-equality failures rather than lowering the expected trial count. It verified all 135 native evidence files for the 45 retained trials, with no additional hash, counter, evidence, per-episode bound, or cleanup-record discrepancies. A separate prefix audit confirms the exact first 45 entries of the frozen execution order, unique identities, contiguous positions, and no selective retries. Combined with regression, 423 native evidence files were verified.

The frozen generic `tasks_completed_both_attempts` metric can count a singleton on partial runs. Correctly requiring both attempts gives three fresh tasks per arm; use coverage-accounting.json (historical evidence omitted) for that field. The generic `selection_eligible` field is not a completed-cohort acceptance gate. No model selection is made from this partial run.

The six new families were authored and independently reviewed before any parser, policy, teacher or model calls. Their novelty inventory covers 448 prior tasks and 92 roots across 16 prior manifests. All 12 authored endpoint fixtures passed the native verifier, and their missing-write negative fixtures failed. The cohort shares known billing primitives and is not an external benchmark. Its native verifier does not fully grade reporting prose, and harness target IDs bypass unrestricted entity discovery. After this run these families are exposed and must never be relabeled fresh.

Source, task, checkpoint, calibration and configuration hashes stayed fixed during evaluation. The retained regression audit remains byte-identical to its frozen hash. New orchestration/report/cohort tests passed: 43 + 21 + 3 = 67 tests. Existing repaired-runtime validation was 668 passed, two skipped, plus 11 Torch tests. Ruff and diff whitespace checks passed for this work.

Detailed per-family outcomes, native results, latency and proposal traces are in results.md (historical evidence omitted), metrics.json (historical evidence omitted), audit.json (historical evidence omitted), and regression-audit.json (historical evidence omitted). Runnable execution, fixture and offline-audit commands are in reproduce.md (historical evidence omitted).

The next change should repair the four parser semantics, add compositional adversarial tests around each grant/prohibition/withdrawal boundary, and replay these now-exposed families before another prospective holdout. Preserve operation-specific restrictions and evidence-based completion. Retraining remains deferred: it cannot restore writes denied by the shared guard, though recovered duplicate-review errors remain a later efficiency target.

[Aggregate results-only metrics](decision-model-generalization-v3/research-metrics.json).

[Public setup and historical-input prerequisites](../experiments/decision_model/README.md).
