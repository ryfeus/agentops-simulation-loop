> Public research summary. Private execution manifests, raw trajectories, checkpoints, operational receipts, and source revisions are omitted. Results below are historical measurements, not independently reproduced by this export.

# Escalation-history follow-up

Exposing actual escalation history prevented duplicate-escalation attempts in this targeted evaluation. It did **not** make the rule/LLM hybrid ready for default use: both modes still performed six unwanted writes because the existing intent parser does not reliably scope authorization to individual invoices. Default agent behavior and the original fine-tuning results remain unchanged.

## Measured results

The same 24 tasks were evaluated once with history hidden and once with history visible. They cover eight newly authored families, with three related variants per family.

| Metric | History hidden | History visible |
|---|---:|---:|
| Policy completed and native verifier passed | 7/24 | 21/24 |
| Native verifier passed, regardless of completion | 8/24 | 21/24 |
| Terminal database-state invariants passed | 21/24 | 21/24 |
| Duplicate-escalation attempts | 12 | 0 |
| All forbidden attempts | 18 | 6 |
| Successful unwanted writes | 6 | 6 |
| Successful expected operations | 30 | 30 |
| LLM calls | 48 | 49 |
| Median policy latency | 6.537 s | 6.589 s |

The 14 improved completion outcomes comprise 12 avoided duplicate escalations, one recovered required inspection, and one difference caused by a provider timeout. There were no paired regressions, seven tasks completed in both modes, and three completed in neither. The timeout occurred in the hidden-history arm after the required writes had succeeded; its native verifier passed, but the policy did not finish. It is retained as a failure, not attributed to the history feature.

Terminal-state results did not improve. A rejected duplicate write can fail trajectory checks while leaving the final database correct. This distinction explains much of the apparent completion gain. Neither LLM usage nor measured latency improved.

## Remaining failure

All three variants of `history-v2-scoped-audit-with-active-refund` failed in both modes. Their request requires inspecting four invoices, leaving the first two unchanged, refunding the third, and confirming the fourth's earlier refund. The original parser applies refund intent across invoices: it refunds the read-only paid invoice and escalates the read-only disputed invoice. Those two unwanted writes per task succeed because the existing billing guard does not enforce invoice-specific user authorization.

These are real writes to isolated synthetic trial databases. Visible escalation history cannot repair this authorization failure. The next implementation should establish explicit invoice-scoped intent and defer ambiguous writes before another deployment decision.

## Implementation and experiment controls

The opt-in tool observation returns only the requested invoice's actual escalation IDs, invoice IDs, and a completeness flag, read from the live SQLite ledger. Missing or malformed history remains unknown. A narrow rule guard avoids escalating an already escalated dispute and defers when required history is unavailable. An existing escalation does not prohibit a separately requested first refund of a paid invoice. Refund history remains unobserved.

The original intent parser, LLM system prompt, native verifier, billing service, tool budget, and fallback configuration were preserved. Both rules and the existing fallback receive the enriched observation. This treatment therefore measures history access plus the narrow duplicate guard; it does not isolate either component. The frozen fine-tuned model was not reused with this changed input contract, and no additional training occurred.

The cohort and runtime sources were frozen before paid evaluation. Trial order alternated which mode ran first, with one attempt per mode/task and no retries. Independent review verified all 48 trial identities and source/task hashes, and all 78 visible-history responses matched the actual target ledger at observation time. No future records, other invoices, or evaluator expectations entered observations.

These are authored, targeted stress cases, not an external blind benchmark. Eight families provide the meaningful grouping; the 24 related variants are not independent samples. Comparison against 340 historical scenarios found no structural or normalized-instruction collisions, which does not establish semantic independence. The three refund-ledger/status-drift component controls also contain a separate already-escalated dispute: their improved whole-task results demonstrate duplicate avoidance on that dispute, not access to hidden refund history. Clarification prose was not scored in the held-authorization family.

[Aggregate results-only metrics](decision-model-history/research-metrics.json).

[Public setup and historical-input prerequisites](../experiments/decision_model/README.md).
