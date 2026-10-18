> Public research summary. Private execution manifests, raw trajectories, checkpoints, operational receipts, and source revisions are omitted. Results below are historical measurements, not independently reproduced by this export.

# Fresh-family evaluation of fallback reason and confidence repairs

**The fixes improved executed work, but did not improve strict completion for the released decider.** Its completion stayed at 8/24, while LLM-only improved from 4/24 to 7/24. No weights were trained or recalibrated. All 96 trials finished within the one-hour cap, with zero unwanted writes, execution errors, or indeterminate tool outcomes.

## Paired results

The prospective cohort has six new composition roots, two presentation variants per root and two attempts per task. Old and repaired fallback runtimes were interleaved using the same released checkpoint, existing calibration, authorization guard, underlying billing service, and budgets. The old runtime was loaded from its exact retained source, not reconstructed. No failed policy trial was retried and no runtime or task changed after the freeze.

| Policy | Strict completion, old → repaired | Native verification, old → repaired | Required writes, old → repaired | LLM calls, old → repaired |
|---|---:|---:|---:|---:|
| Released decider | 8/24 → 8/24 | 10/24 → 12/24 | 36/76 → 40/76 | 131 → 134 |
| LLM-only | 4/24 → 7/24 | 4/24 → 9/24 | 29/76 → 36/76 | 195 → 205 |

Strict completion requires both the policy’s resolved-invoice check and the native state/trajectory verifier. A native pass with an unresolved invoice remains a failure. Neither metric establishes the quality of clarification or reporting prose. The repaired LLM-only arm won three matched trial pairs and lost none; all three gains were in one family. The released arm had no strict-completion wins or regressions. Six closely related families do not establish broad generalization.

See full per-family metrics and latency (historical evidence omitted), paired trials (historical evidence omitted), and machine-readable metrics (historical evidence omitted).

## Implemented changes

- Invoice-local LLM write tools now supply the existing application reason by default. Their schemas and descriptions state that a separate customer-provided reason is unnecessary for an otherwise authorized action.
- Trusted continuation context explains that decision uncertainty neither grants nor revokes request authority. Raw routing labels such as `low_confidence` and `defer` stay in diagnostic telemetry rather than being presented as permission signals.
- Invoice scope and request authorization remain enforced at the execution boundary. The historical whole-task tool schema still requires a reason; default application behavior and billing service policy are unchanged.

Actual graph and Bedrock schema/message serialization tests cover omitted reasons, explicit denials, legacy schemas, and confidence wording. The model itself is unchanged: all 126 exact observed-state pairs produced identical actions and probabilities. Model comparison (historical evidence omitted).

## Why completion remains low

The frozen authorization parser fails to recognize some valid request forms in the fresh families. It fails to recover some invoice-specific grants or resolve their conditions, then denies the required action. Across all arms, 144 of the 163 missing writes involved absent parsed authority: 112 were attempted and rejected, and 32 were never attempted. All 36 missing writes in the repaired released-decider arm fall in this group. The decision model and fallback cannot execute past that guard. The recorded failures involve an unrecognized existing-case condition (64 omissions), a specialist-review/customer suffix that loses the escalation grant (48), and an absent-invoice condition that contaminates independent scopes (32). These are source-based diagnoses, not rerun counterfactuals.

A separate completion problem affects preserved-review inspections. The parser leaves their no-write scope ambiguous, so the episode remains unresolved even when required writes and native state checks pass. There are 32 such unresolved invoice records across the arms; the repaired released arm has four trials that pass native verification but fail strict completion.

The remaining 19 missing operations had parsed authority and were not attempted. Exact responses and post-hoc categories are retained in failure analysis (historical evidence omitted); manual adjudication (historical evidence omitted) separates the targeted reason/confidence failures from other omissions.

Manual review found 15 omitted writes with parsed authority in the old arms: ten cases misread `defer` as withheld permission, three misread low confidence, and two primarily asked for the tool reason. The repaired arms had four such omissions, all LLM-only: one demanded an unavailable refundability field, two demanded an unnecessary dispute prerequisite, and one asked for redundant refund confirmation. None of the 17 repaired-arm unattempted writes reviewed (including 13 without parsed authority) repeated the targeted tool-reason or routing-label mistake. This is a scoped observation, not proof that all unnecessary clarification has been eliminated. All 163 missing operations followed fallback; none followed an accepted direct `no_action`.

Independent semantic review (historical evidence omitted) found the expected writes authorized and service-feasible. It also found a shared tool-description mismatch: escalation is described as applying to disputed invoices, while the service permits the explicit paid-invoice review requests in this cohort. This is an additional plausible source of refusals, not an identified causal effect of the two repairs.

## Freshness and interpretation

The cohort was frozen before any policy or model rollout. Root, behavioral-structure and normalized-instruction checks found no collisions against 412 prior tasks / 74 prior roots spanning all previous splits. Expected writes were authored independently of the authorization parser; offline endpoint fixtures validated the unchanged native verifier. No training examples, policy preflight, selection, or calibration used the new cohort. Cohort design and manifest (historical evidence omitted).

These are author-constructed compositions in the same billing domain, created after prior failure analysis. New structural signatures are not proof of semantic independence. Variants and repeated attempts are correlated, both fixes changed together, and provider responses can vary at temperature zero. The previous exposed-cohort success rates are not directly comparable with these fresh tasks. Further changes must treat this cohort as exposed.

[Public setup and historical-input prerequisites](../experiments/decision_model/README.md).
