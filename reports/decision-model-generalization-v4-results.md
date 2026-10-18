> Public research summary. Private execution manifests, raw trajectories, checkpoints, operational receipts, and source revisions are omitted. Results below are historical measurements, not independently reproduced by this export.

# Parser v7: regression repaired, fresh completion still limited

The four measured authorization defects are repaired. All 72 paid regression trials passed, including the previously failing v3 tasks. The fresh comparison completed all 48 planned trials, but the released decider and LLM-only each completed **8/24 (33.3%)**. There were **zero unwanted writes in all 120 trials**. No training or recalibration occurred.

## Measured outcomes

| Cohort and arm | Strict completion | Native passes | Required writes | Unwanted writes |
|---|---:|---:|---:|---:|
| Exposed regression, released | 36/36 | 36/36 | 96/96 | 0 |
| Exposed regression, LLM-only | 36/36 | 36/36 | 96/96 | 0 |
| Fresh v4, released | 8/24 | 20/24 | 48/52 | 0 |
| Fresh v4, LLM-only | 8/24 | 20/24 | 48/52 | 0 |

All 24 fresh task/attempt pairs tied: no released-model wins or losses. Each arm completed both attempts on four of the 12 tasks, covering two of six families. The customer read-only boundary and partial-withdrawal/status-drift families passed every trial; the other four families failed every trial. There were no provider/inference execution errors, missing trial records or selective retries.

The matched first-attempt v3 subset improved from **4/12 to 12/12 per arm**, eight wins and no losses. The older matched regression subset remained 24/24 per arm. These are previously exposed engineering comparisons against historical runs, not fresh or randomized evidence of a learned improvement. Fresh v4 and prior v3 scores must not be compared as if they used identical tasks.

## Why fresh trials still fail

The 32 strict failures divide into four groups of eight trials, including both arms and attempts:

| Cause | Observed consequence | Correct next change |
|---|---|---|
| Pending refund approval contaminates independent review authority | An explicitly requested review is blocked despite a matching invoice and complete empty review history; eight required reviews are missing | Represent uncertainty per operation, so a refund condition does not revoke an independent review grant |
| Unrecognized existing-review phrasing | “No escalation is already recorded for …” is not recognized; complete matching history shows the review, but completion remains unresolved | Normalize equivalent review conditions and derive completion from observed history |
| Recorded-refund confirmation lacks evidence | Paid status plus empty escalation history cannot confirm a historical refund; database state passes but the confirmation remains unresolved | Expose target-matched refund-ledger evidence and define evidence-based confirmation |
| Conditional refund lacks evidence | The fallback assumes no prior refund from empty escalation history and attempts a refund; the guard correctly blocks it | Require complete refund history to establish absence; teach the fallback to report unavailable evidence accurately |

The last two groups expose an observation-interface limitation, not a reason to relax authorization. These authored tasks have valid database endpoints, but the current observation interface cannot establish their refund-history conditions. All 16 corresponding fallback responses made unsupported assertions about absent prior refunds. Native verification does not fully grade prose, so a native pass is not proof of a truthful response.

Manual review found **11 genuine released-model proposal errors in regression and six in fresh evaluation**, all recovered by routing/guard behavior. Of eight fresh proposals automatically labeled prohibited, two were actually legitimate reviews rejected by the parser; they must not be counted as model errors. Zero executed unwanted writes does not mean there were no unsafe proposals. See the full trace diagnosis (historical evidence omitted) and machine-readable diagnosis (historical evidence omitted).

## Efficiency and resources

| Cohort and arm | Bedrock calls | Input tokens | Output tokens | Policy latency p50 / p95 |
|---|---:|---:|---:|---:|
| Regression, released | 178 | 332,433 | 21,385 | 19.25 / 26.11 s |
| Regression, LLM-only | 339 | 644,031 | 37,612 | 19.74 / 24.55 s |
| Fresh, released | 128 | 243,429 | 16,322 | 18.24 / 20.99 s |
| Fresh, LLM-only | 184 | 353,080 | 21,874 | 17.46 / 18.73 s |

The released model used **30.4% fewer provider calls on fresh tasks**, but its median policy latency was higher and completion was identical. Calls and returned token usage are measured; dollar charges were not queried and are not inferred. Policy latency excludes environment setup, verification and cleanup.

Paid execution used local MPS and Bedrock `us.anthropic.claude-sonnet-4-6`, with refreshed AWS `default` credentials verified for a private research account, region `us-west-2`. Regression took **1,892.585 seconds**, fresh evaluation **1,163.686 seconds**: **3,056.272 seconds (50m 56s)** combined, within the 3,600-second bound. No EC2 training worker was provisioned, training steps were zero, and calibration fits were zero. All 120 trial environments were recorded as deleted; the final independent Docker inventory matched the four preexisting containers exactly, with no additions or removals.

[Aggregate results-only metrics](decision-model-generalization-v4/research-metrics.json).

[Public setup and historical-input prerequisites](../experiments/decision_model/README.md).
