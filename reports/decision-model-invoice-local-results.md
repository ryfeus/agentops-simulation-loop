> Public research summary. Private execution manifests, raw trajectories, checkpoints, operational receipts, and source revisions are omitted. Results below are historical measurements, not independently reproduced by this export.

# Invoice-specific authorization and invoice-local fallback

Implemented and evaluated the repairs before any further training. **No weights were trained or recalibrated.** The released checkpoint and previously selected epoch-0 checkpoint, original calibration, fallback model/configuration, wheel and corpus are unchanged. The raw `tuned` arm is tensor-identical to the released baseline; it is not a newly trained model.

## Paired native results

The same 12 exposed tasks, six composition roots and two attempts per arm were used. The comparison below pairs all 96 final trials with the previous recovery-adjusted run by task, arm and attempt. These are engineering regressions on inspected cases, not fresh held-out evidence or a claim of learned improvement.

| Arm | Completed before → after | Required writes before → after | Unwanted writes before → after | LLM calls before → after | Improved / regressed pairs |
|---|---:|---:|---:|---:|---:|
| Scoped rule | 12/24 → 24/24 | 36/64 → 64/64 | 2 → 0 | 20 → 0 | 12 / 0 |
| Released decider | 12/24 → 24/24 | 36/64 → 64/64 | 0 → 0 | 72 → 92 | 12 / 0 |
| Selected epoch-0 baseline | 12/24 → 21/24 | 36/64 → 61/64 | 0 → 0 | 72 → 89 | 11 / 2 |
| LLM-only | 12/24 → 18/24 | 36/64 → 54/64 | 0 → 0 | 72 → 194 | 6 / 0 |

Completion requires both policy completion and the independent native state/trajectory verifier. Unresolved invoices, omitted required writes and blocked attempts are not successes. No failed final policy trial was retried.

| Arm | Policy latency p50 | Policy latency p95 | Fallback continuations | Unresolved invoices |
|---|---:|---:|---:|---:|
| Scoped rule | 2.20 s | 2.99 s | 0 | 0 |
| Released decider | 17.28 s | 25.03 s | 82 | 0 |
| Selected epoch-0 baseline | 16.89 s | 23.47 s | 82 | 3 |
| LLM-only | 22.99 s | 29.53 s | 140 | 10 |

Policy latency excludes Docker setup, verification and cleanup. Repeated attempts are correlated. See full per-family results (historical evidence omitted), matched comparison (historical evidence omitted), and metrics (historical evidence omitted).

## Residual failures

The final run omitted 13 required operations: 12 escalations because fallback asked for a reason instead of calling the authorized tool (10 LLM-only, two selected-baseline), and one selected-baseline refund because fallback interpreted `low_confidence` as missing permission. All missing operations had observed request authority. Direct actions receive the existing standard reason automatically, while the LLM tool interface requires a reason argument before the wrapper normalizes it. These are recorded fallback/interface failures; their diagnosis is post hoc, not a counterfactual causal test. See failure evidence (historical evidence omitted).

The released and selected checkpoints produced identical actions and probabilities across all 126 pairs with identical observed states. All three selected-baseline failures followed correct proposals matching the released arm; the divergence occurred in fallback responses. The 24/24 versus 21/24 result is therefore not evidence of a learned checkpoint difference. See model pair check (historical evidence omitted).

Completion improved, but model-assisted execution used more LLM calls: released 72 → 92 and LLM-only 72 → 194. This is not a blanket efficiency improvement.

## What changed

- Authorization v3 scopes commands, withdrawals, read-only exceptions and absent-invoice reporting to their invoice and operation. State-conditional disputed-refund and duplicate-review prohibitions no longer deny independent authorized work. Preserving an existing review does not revoke a separately authorized first refund. Unknown language and unresolved conditions still grant nothing.
- Each fallback receives trusted system scope, the original user request and only its current invoice’s observations. The execution wrapper rejects cross-invoice reads and writes and rechecks request authority before every write. Other invoices continue after a local error or unresolved decision.
- The episode shares 16 tool calls and 16 LLM starts, with one continuation per invoice and a 120-second continuation cap. Direct and fallback no-action completion require supporting observed scope. The final job had a 3,600-second cap, 180-second per-task policy/startup cap, and separate 30-second verifier and cleanup caps.
- Scoped authorization replaces the legacy whole-request risk flag in the opt-in runner. Cancellation retains completed actions and known usage; interrupted tool outcomes are marked indeterminate. The default application agent, historical experiment artifacts and billing service policy are unchanged.

## Disclosed interrupted integration attempt

The first invoice-local runtime revision put the routing directive in a new user message. In native execution, the fallback misidentified that legitimate harness directive as prompt injection and attempted other invoices. The scope guard blocked these calls, but they exhausted the shared tool budget. The attempt was interrupted after six trials started; five final records and a sixth cancellation snapshot remain intact in `.artifacts/decision-model-invoice-local/final-harbor/`.

The follow-up revision moved the harness contract into system context and instructed the fallback not to infer the status of other invoices. A test exercises the actual graph and Bedrock serializer to verify those roles. A new freeze and new output directory were used for the full 96-trial rerun. This is a disclosed runtime iteration on an exposed cohort, not an infrastructure-only recovery or a single untouched preregistered experiment. Models, calibration and task expectations were never changed. See initial-attempt accounting (historical evidence omitted).

[Aggregate results-only metrics](decision-model-invoice-local/research-metrics.json).

[Public setup and historical-input prerequisites](../experiments/decision_model/README.md).
