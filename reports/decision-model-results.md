> Public research summary. Private execution manifests, raw trajectories, checkpoints, operational receipts, and source revisions are omitted. Results below are historical measurements, not independently reproduced by this export.

# Strands Decider billing continuation experiment

Executed 2026-10-16. Fine-tuning worked technically and improved the fixed replay set, but the deterministic rule remained the strongest measured option. The tuned hybrid saved 3 of 21 LLM calls without improving task success or latency. Present this as an exploratory result about the limits of adapting a decision model to a largely structured policy; do not enable it in the default agent.

| Question | Measured answer |
|---|---|
| Does adaptation improve the released model? | Yes on this diagnostic: 66/150 → 137/150 correct. Initial invoice inspections improve only 43/60 → 47/60. |
| Do structured rules already solve the useful part? | Largely: rule accuracy is 144/150 overall and 54/60 on initial inspections. |
| Does the tuned hybrid help the agent? | Same 9/12 completed-and-verified tasks as the existing LLM continuation; 21 → 18 LLM calls, but median policy latency 3,519 → 3,804 ms. |
| What performed best? | Rule plus existing LLM fallback: 9/12 completed-and-verified, nine LLM calls, median policy latency 743 ms. |

The implementation is opt-in under experiments/decision_model (historical evidence omitted). Default application behavior, billing enforcement, historical experiments, original checkpoint, and application/RL lockfiles were preserved. No service was deployed and no changes were pushed.

## Data and evidence boundary

Recovered 160 verified Phase 12 demonstrations from the local evidence archive. Extraction produced 480 four-option examples: 185 actual inspection snapshots, 110 actual completed-action histories, and 185 explicitly marked missing-observation ablations. The model sees only the request, observed target identity, policy, observations and prior completed actions at the decision boundary. Task IDs, family names, expected outcomes and future observations remain metadata.

All source scenarios were historical **training** scenarios. Sixteen source prototypes were assigned before augmentation; historical development/test splits were not trained on or repurposed. These are newly held-out training families, not an independent external benchmark.

| Partition | Prototypes | Rows | Refund | Escalate | No action | Defer |
|---|---:|---:|---:|---:|---:|---:|
| Train | 8 | 225 | 35 | 10 | 85 | 95 |
| Development selection | 2 | 60 | 15 | 5 | 20 | 20 |
| Calibration | 2 | 45 | 10 | 5 | 15 | 15 |
| Final test | 4 | 150 | 15 | 15 | 50 | 70 |

The test contains 40 tasks, 60 initial inspections, 30 completed-action snapshots and 60 missing-observation ablations. Exact visible-state duplicate/conflict checks found 480 unique states and no cross-split duplicates. An audit checked extraction against all source observations. These checks do not establish semantic independence: related policy patterns cross prototypes, and all 30 completed-action test snapshots belong to one prototype.

`get_invoice` does not expose pre-existing escalation records. Fifteen teacher no-ops therefore become `defer`, including ten in the test. Different request wording can still conceal identical uncertainty. The dataset cannot establish that a model inferred hidden history, or that escalation is universally safe. Evaluation-only task permissions classify forbidden attempts after execution and never enter the model input.

## Training and calibration

Pinned sources:

- Strands Decider: `6039cf8b64c6074e75804000c07ac3d4cce795f7`.
- Released checkpoint: `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`, revision `216d1ccb46a9e9b036eb9187b248db98480f77d4`.
- Base: `Qwen/Qwen3.5-2B-Base`, revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`. Upstream describes this historical base revision as inferred; we pin its published provenance without strengthening that claim.

The upstream `init_from` path was confirmed to replace the head, so continuation explicitly preserves the released rank-48 LoRA and 768-dimensional pointer head. The wrapper reuses upstream rendering, collation, option positions and supervised choice loss. Trainable parameters: 50,457,600 LoRA plus 3,151,360 head; 1,881,825,088 base parameters frozen. Training resets inference temperature and shuffles option ordering with label remapping. The source checkpoint stays unchanged.

The 20-step smoke test passed: finite adapter/head gradients, changes to both, frozen base, and exact probability round trips on four balanced fixtures before and after updates. Fixture accuracy rose 50% → 100%; this demonstrates wiring, not quality. Smoke work took 59.93 seconds.

The single main configuration used seed 42, AdamW, LoRA LR 1e-5, head LR 1e-4, microbatch two, accumulation four, maximum length 1,024 with truncation rejected, and gradient checkpointing. It completed **three epochs / 87 optimizer steps**, below the 200-step cap. Training including development checks took 150.78 seconds, peaking at 5,013,767,680 bytes allocated CUDA memory. This is not the billed worker occupancy.

Checkpoint selection minimized observed-policy prohibited proposals, then maximized development accuracy, retaining the earlier candidate on ties. All three epochs achieved 60/60 with zero observed-policy prohibited proposals; **epoch 1, step 29** was selected. The main run restarted from the release, not from the overfit smoke checkpoint. `best/` and `last/` are retained separately.

On the reserved 45 calibration examples, temperature 1.5 reduced NLL 0.05351 → 0.04879. A threshold of **0.9951937004412189** accepted 28/45 decisions without observed accepted errors. Temperature and threshold use the same sparse two-family calibration partition; zero observed errors is not a safety guarantee. The untuned hybrid was separately calibrated on these same reserved examples (threshold 0.7265781481220226; five accepted). No final-test feedback changed weights, selection or thresholds.

## Decision replay

Each arm evaluated the same 150 fixed inputs once. The LLM arm made 150 actual Sonnet 4.6 Bedrock requests with the same observations and four choices. No LLM probabilities were invented. Prohibited proposals below mean contradictions detectable from the supplied observation/policy, before any execution guard.

| Arm | Correct / 150 | Accuracy | Macro F1 | Prohibited proposals | Defer predictions |
|---|---:|---:|---:|---:|---:|
| Rule | 144 | 96.0% | 0.927 | 0 | 76 |
| Released decider, L4 | 66 | 44.0% | 0.432 | 24 | 17 |
| Tuned decider, L4 | 137 | 91.3% | 0.893 | 2 | 65 |
| Existing LLM, constrained choice | 102 | 68.0% | 0.652 | 0 | 32 |

All arms had zero replay errors. Defer frequency is not task success and differs from calibrated hybrid routing, which also falls back on low confidence.

| Accuracy slice | Rows | Rule | Untuned | Tuned | LLM |
|---|---:|---:|---:|---:|---:|
| Actual observed states | 90 | 93.3% | 54.4% | 85.6% | 84.4% |
| Initial invoice inspection | 60 | 90.0% | 71.7% | 78.3% | 76.7% |
| Completed action history | 30 | 100% | 20.0% | 100% | 100% |
| Missing-observation ablation | 60 | 100% | 28.3% | 100% | 43.3% |

Of the 71 additional correct tuned predictions, 43 come from artificial missing observations, 24 from completed-action history, and **four from initial inspections**. Prototype-cluster bootstrap intervals (2,000 samples, seed 17) for tuned-minus-untuned accuracy are +47.3 percentage points [24.0, 59.0] overall, +31.1 [-20.0, 48.4] on observed states, and +6.7 [-20.0, 20.0] on initial inspections. Four clusters cannot support strong generalization claims. Fixed four-class macro F1 on a single-class slice has a maximum of 0.25; use slice accuracy for that comparison.

Warm batch-one L4 forward latency was 66.81/68.62 ms p50/p95 untuned and 65.55/67.58 ms tuned; model load was separately 14.61 and 4.30 seconds. The rule measured a local Python function; LLM 809.54/897.08 ms measured full Bedrock requests including transport. These boundaries and devices differ; this comparison does not measure end-to-end savings. Input-length distributions, per-class recall, calibration and all intervals are in metrics.json (historical evidence omitted) and metrics.csv (historical evidence omitted). All-row decider ECE improved 0.220 → 0.062, combining adaptation and recalibration: untuned replay uses the released temperature 0.86296, while tuned replay uses billing-calibrated temperature 1.5. Rule one-hot scores are deterministic outputs, not empirically calibrated confidence.

Observed-state quality and separately scoped latency (historical evidence omitted)

Confusion matrices (historical evidence omitted)

## Actual Harbor execution

The opt-in runner resolves targets, performs a fixed inspection stage, and uses the existing billing tools and native Harbor verifier. This is a **constrained hybrid policy**, not the original agent from the beginning of a request. Its existing `create_agent` LLM continuation receives the same tool history. All comparable local arms use the same 12 tasks from four prototypes, one attempt each, enforced policy, 16-tool budget and reason formatting. Model instances are reused. Fallback occurs at most once per task, then the existing LLM handles remaining work; its internal calls are counted.

Local timings use Apple M1 Max / 32 GiB / MPS and Docker 28.5.1. Policy latency includes decisions, fallback and tools, excluding environment startup, verifier and cleanup. Models load separately (tuned 10.47 seconds). These measurements are separate from L4 replay.

| Local arm | Completed and verified | LLM calls | Tasks with fallback | Forbidden attempts / successful writes | Median policy ms |
|---|---:|---:|---:|---:|---:|
| Existing LLM continuation | 9/12 | 21 | 12 | 3 / 0 | 3,519 |
| Rule + LLM | 9/12 | 9 | 4 | 3 / 0 | 743 |
| Untuned + LLM | 9/12 | 21 | 12 | 3 / 0 | 4,200 |
| Tuned + LLM | 9/12 | 18 | 12 | 3 / 0 | 3,804 |

Tuned accepted three refund decisions before later fallback on the same composed tasks, saving one LLM call each. Every task still reached fallback. The 14.3% call reduction did not produce a latency win; the rule hybrid reduced calls by 57.1%. The 9/12 successes were the same tasks across all four arms, with identical failures on hidden prior escalations. One attempt per task does not quantify stochastic variability.

Standalone rule execution had 11/12 verifier passes but only 8/12 explicitly completed tasks: four deferrals had no configured fallback, and already-satisfied state could pass verification despite unresolved policy control. The same distinction applies to standalone model runs. At the tested operating points, standalone untuned (threshold 0, released calibration) completed and verified 7/12, with three rejected forbidden attempts; standalone tuned (reserved-data threshold 0.99519) had 9/12 verifier passes but **zero explicitly completed tasks**, because all twelve requested unavailable fallback. These use different routing thresholds and are not a controlled comparison of raw model quality. Full standalone GPU results and timing boundaries are in harbor-standalone.json (historical evidence omitted); these runs make no LLM calls. Additional full task timings, tool errors and successful-tasks-per-minute accounting are in harbor-timing.json (historical evidence omitted). Do not pool local and GPU timings.

## Remaining failures and recommendation

- Tuned proposed two refunds on requests explicitly asking for no changes (confidence approximately 0.596). The chosen hybrid threshold rejects these proposals; the billing service's disputed-invoice guard alone is not protection against every unwanted paid-invoice refund.
- Tuned escalated eight of ten hidden-history examples labeled defer. These are absent from the observed-policy prohibited count because the input does not reveal the pre-existing escalation. Harbor caught rejected duplicate attempts; the native trajectory verifier failed all three such sampled tasks even though no forbidden write succeeded.
- Reversing option order changed 3/12 untuned and **6/12 tuned** predictions. Those six tuned changes were hidden-history escalation → defer. This subset is concentrated in one family, but demonstrates real position sensitivity despite correct label remapping and training shuffles.
- A correct read-only/no-action proposal can remain below the conservative threshold and fall back, eliminating call savings. Composed tasks often accept one decision and still need the LLM for another invoice.

The useful next step is to make existing escalation history observable and re-evaluate the rule-plus-LLM policy on genuinely independent families before another training campaign. The measured evidence supports a meetup demonstration of functioning continuation training and why a strong rule baseline matters. It does not justify enabling this learned policy by default.

[Aggregate results-only metrics](decision-model/research-metrics.json).

[Public setup and historical-input prerequisites](../experiments/decision_model/README.md).
