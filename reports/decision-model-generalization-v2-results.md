> Public research summary. Private execution manifests, raw trajectories, checkpoints, operational receipts, and source revisions are omitted. Results below are historical measurements, not independently reproduced by this export.

# Authorization and fallback repair: regression and prospective evaluation

**Fresh-family strict completion: released decider 20/24; LLM-only 20/24.** No weights were trained or calibration fitted. Both arms use the same repaired runtime and Bedrock fallback. The decider arm retains the released checkpoint and routing threshold; LLM-only bypasses it.

## Exposed regression

The repair comparison uses the same 12 generalization-v1 tasks, exactly attempt 0 for each arm before and after repair. These tasks were already exposed and informed the repairs. The comparison is a regression check, not fresh generalization evidence; provider variation and combined repairs prevent isolating an individual fix's effect.

| Arm | Before strict | After strict | Before required writes | After required writes | Before / after unwanted writes | Improved / regressed / tied tasks |
|---|---:|---:|---:|---:|---:|---:|
| released | 3/12 | 10/12 | 22/32 | 32/32 | 0 / 0 | 7 / 0 / 5 |
| llm | 3/12 | 12/12 | 22/32 | 32/32 | 0 / 0 | 9 / 0 / 3 |

| Repaired regression arm | Native passes | Missing writes | Authorization-blocked attempts | Scope-blocked attempts |
|---|---:|---:|---:|---:|
| released | 12/12 | 0 | 6 | 0 |
| llm | 12/12 | 0 | 0 | 0 |

Regression coverage and strict model quality are distinct: required writes may complete while rejected duplicate proposals or other execution errors still fail strict completion. The regression gate's recorded scope is disclosed in the runtime lock and validation evidence; its integrity audit does not certify all policy outcomes.
Core regression gate passed: True. All strict-quality trials passed: False. Retained strict failures: 2. The core gate requires native passes, all required writes, no extra writes, no scope violations and complete execution evidence; it does not waive strict failures from the reported scores. The audit retains their proposal and error details.

## Prospectively held-out families

The fresh design is 12 tasks in 6 composition roots, two variants per root and 2 attempts per arm/task: 48 retained native trials. The authored tasks require 34 new writes per 12-task pass. No fresh-cohort parser, policy, teacher or model runs preceded the freeze. Independent semantic review and offline endpoint fixtures establish authored consistency, not policy performance.

| Arm | Strict completion | Native passes | Required writes | Unwanted writes | Both attempts complete |
|---|---:|---:|---:|---:|---:|
| released | 20/24 | 24/24 | 68/68 | 0 | 10/12 |
| llm | 20/24 | 24/24 | 68/68 | 0 | 10/12 |

Matched fresh task/attempt outcomes: released wins 0, loses 0, ties 24. Strict completion requires both policy resolution and native state/trajectory verification. Scope/tool errors remain failures even if endpoint invariants pass. Reporting and clarification prose are not fully scored; a native pass is not complete prose validation.

| Fresh composition root | Released strict | LLM strict | Missing writes (released / LLM) | Unresolved invoices (released / LLM) |
|---|---:|---:|---:|---:|
| generalization-v2-ledger-asymmetry-with-missing-reference | 4/4 | 4/4 | 0 / 0 | 0 / 0 |
| generalization-v2-same-status-three-owner-action-grid | 4/4 | 4/4 | 0 / 0 | 0 / 0 |
| generalization-v2-two-absences-and-three-customer-actions | 4/4 | 4/4 | 0 / 0 | 0 / 0 |
| generalization-v2-two-households-cross-operation-history | 0/4 | 0/4 | 0 / 0 | 8 / 8 |
| generalization-v2-two-withdrawals-preserve-separate-grants | 4/4 | 4/4 | 0 / 0 | 0 / 0 |
| generalization-v2-unselected-cross-operation-pair | 4/4 | 4/4 | 0 / 0 | 0 / 0 |

Recorded fresh missing-operation groups: no missing required writes. These describe trace evidence, not isolated causal effects. A guard-prohibited proposal or blocked attempt is not proof the user prohibited it; the guard can over-deny. Successful unwanted writes are measured against authored expectations, separately from rejected proposals.

Fresh metrics and latency (historical evidence omitted); paired records (historical evidence omitted); residual evidence (historical evidence omitted).

[Public setup and historical-input prerequisites](../experiments/decision_model/README.md).
