> Public research summary. Private execution manifests, raw trajectories, checkpoints, operational receipts, and source revisions are omitted. Results below are historical measurements, not independently reproduced by this export.

# Larger-worker continuation experiment

Completed: training, held-out replay, 96 native trials plus four credential-only recoveries, independent audits, and AWS/container cleanup.

A paid `g6e.16xlarge` with one NVIDIA L40S was allocated in `us-west-2a` after the larger-capacity search. **A new decision model was trained, but the frozen development selector rejected both trained candidates and retained the released baseline.** There is no demonstrated fine-tuning gain.

## Training and selection

The known authorization regression repair passed all 24 previously inspected tasks with all 30 required writes and no unwanted writes, as recorded in the preceding report (historical evidence omitted). That result did not establish correctness for unseen request wording.

Historical official DEV and TEST families were excluded from training. Previously inspected history tasks belong to TRAIN; newly authored composition roots were split before variants. These partitions share billing primitives and are not independently authored benchmarks.

The fresh corpus and training specification were unchanged: one seed (42), the released rank-48 adapter and 768-dimensional pointer head, frozen base weights, microbatch 2 with accumulation 4, and adapter/head learning rates `1e-5`/`1e-4`. One released-policy rollout pass on 48 TRAIN tasks added ten actual reached states to the original 321 training rows. Labels were assigned independently from authored intent and observations, rather than copied from model proposals.

The 20-step smoke passed: balanced-fixture accuracy 0% → 100%, zero probability drift on both checkpoint round-trips, nonzero adapter/head gradients, and frozen base weights. Measured smoke-loop and checkpoint-validation time was 58.89 seconds, including post-training evaluation and save/reload checks. The completed main attempt ran 84 optimizer steps over two epochs, with 144.40 seconds measured from the training loop through candidate evaluation. The training audit passed 43 checks. Direct CPU tensor comparisons confirmed that all six selected-head tensors and all 372 selected-adapter tensors exactly equal the released weights. All six head tensors and all 372 adapter tensors in the rejected `last` checkpoint changed, confirming that training occurred even though selection retained epoch 0. See tensor audit (historical evidence omitted).

| Candidate | DEV replay correct | Completed + verified DEV tasks | Genuine unwanted proposals, posthoc audit | Successful unwanted writes |
|---|---:|---:|---:|---:|
| Released, epoch 0 — selected | 52/71 (73.2%) | 2/8 | 0 | 0 |
| Epoch 1, step 42 — rejected | 66/71 (93.0%) | 1/8 | 0 | 0 |
| Epoch 2, step 84 — rejected | 67/71 (94.4%) | 2/8 | 1 | 0 |

Selection used the predeclared executed-task score, including safety, completion and omissions; replay was only the final tie-breaker. Its recorded accepted-prohibited counts were 2, 5 and 7. **These are not counts of genuine unsafe actions:** the independent DEV-only audit found false flags from a legacy global read-only rule and authorization over-denial. A posthoc diagnostic ranking using genuine unwanted proposals and completion still prefers the released baseline; the frozen selection was not recomputed or changed.

Both released and selected checkpoints calibrated to temperature 1.0 using 157 replay rows and routing threshold 0.7 using 24 executed standalone tasks. Temperature and routing shared the same four-family reserved calibration partition; this is not a safety guarantee, and standalone routing differs from the final hybrid. The selected `training/best` and raw arm named `tuned` represent epoch 0. Trained but rejected weights remain in `training/last`; final comparisons must not present the selected arm as a learned improvement.

## Why replay improved without task success

The DEV failure audit (historical evidence omitted) separates three causes:

- Remaining authorization over-denial blocked four required operations for every candidate. “Create a duplicate review” was not recognized by the negative-clause parser, and a status-only exception mentioning two invoices overrode a separate authorized refund.
- A legitimate deferral on unavailable evidence terminated the standalone task before two independent later operations. Standalone development had no LLM fallback; the final hybrid does.
- Training introduced model errors beyond those harness limits: it omitted an authorized refund after an earlier invoice action, and epoch 2 confidently proposed a refund on a withdrawn authorization. The guard blocked that unwanted write.

Snapshot accuracy therefore overstates the effect on sequential task completion. No policy, risk metric, threshold or model was changed after final selection to accommodate test results.

## Disposition

Do not promote the trained weights. The known 24-task authorization repair succeeded, but unseen phrasing still exposes over-denial and late-withdrawal gaps. A follow-up should first unify invoice-scoped permission logic across the executor, model features, and risk scorer, and test coordinated commands, exceptions, negation and withdrawal explicitly. Independent verified intent should remain the label truth: required work blocked by a faulty parser must not be relabeled as `no_action` merely to agree with the guard. It should also make missing evidence and fallback invoice-local where the task permits independent remaining work. Only then should another training run target the actual omission and ordering failures. The held-out families used here are now exposed and must not be represented as fresh in another run. No such follow-up training or policy change was performed in this frozen experiment.

## Fresh held-out evaluation

Selection completed at 18:45:38 UTC. Artifacts were downloaded and checksum-verified, then the final artifact freeze was written before collecting TEST teachers. The twelve held-out tasks span six fresh authored composition roots, two variants each. All twelve teacher trajectories passed native verification and produced 87 replay rows: 70 initial observations, nine actual completed-action states and eight masked-observation diagnostics. Cohort, scenario and task hashes remained unchanged. These are authored composition holdouts, not an external benchmark.

The frozen scoped rule scored **61/87 (70.1%)** on the same snapshots (rule replay (historical evidence omitted)). On the 70 natural initial snapshots alone, the rule scored 44/70 (62.9%) and the model 37/70 (52.9%); the remaining rows are nine actual completed states and eight masked-observation diagnostics. The offline prohibited-action annotations flagged two rule proposals and five model proposals. These are fixed-state proposal counts, not executed writes, and the legacy risk-annotation caveat still applies. Both rule cases were refunds on the late-withdrawal variants.

Frozen GPU replay scored **44/87 (50.6%)** for both the released model and selected epoch-0 artifact, with no inference errors and exactly identical action probabilities. All inputs fit the 1,024-token cap (maximum 951). All 24 option permutations were checked for 21 family/class representatives: only 15/21 retained one action across every ordering. This is a limitation of the retained baseline, not evidence about the rejected trained weights. GPU forward-pass medians were 67.3 and 63.8 ms; that small timing difference between identical models is not a training gain. See replay summary (historical evidence omitted).

The original 96-trial run completed in 1,529.48 seconds. Four late fallback calls failed with `ExpiredTokenException`. Fresh default-profile credentials were verified, then **only those four exact trial identities** were rerun once in a separate directory, with unchanged tasks, model artifacts, thresholds, budgets and LLM configuration. All four recovered trials passed in 85.08 seconds. Original outcomes remain preserved: rule/released/selected/LLM completion was 12/11/10/11 out of 24. No policy failures were retried.

The following **recovery-adjusted 96-trial view** replaces those four infrastructure-failed records and links every replacement to its original evidence:

| Arm | Completed + verified | Unwanted writes | Missing required operations | LLM calls | Policy latency p50 |
|---|---:|---:|---:|---:|---:|
| Scoped rule | 12/24 | 2 | 28 | 20 | 2.26 s |
| Released decider | 12/24 | 0 | 28 | 72 | 13.25 s |
| Selected epoch-0 baseline | 12/24 | 0 | 28 | 72 | 12.46 s |
| LLM | 12/24 | 0 | 28 | 72 | 10.41 s |

Each arm completed both attempts on six of twelve tasks and executed 36/64 required writes. Both model arms invoked fallback on **all 24 tasks**, so there was no LLM-call saving. The selected checkpoint is the release itself; different timings are run variation, not a fine-tuning effect. Policy latency excludes Docker setup, verification and cleanup.

**The authorization repair is incomplete.** The rule arm successfully refunded the withdrawn invoice in one late-withdrawal variant on both attempts, while denying the still-authorized invoice. The three other arms had zero unwanted writes, but all arms omitted 28 required operations. Zero writes alone would not establish success. Remaining over-denial and withdrawal parsing require repair before deployment.

See the adjusted native report (historical evidence omitted), paired metrics (historical evidence omitted), 100-trial audit and usage (historical evidence omitted), and preserved original report (historical evidence omitted). Repeated attempts are correlated; no claim of broad generalization follows from twelve authored tasks.

## Evidence

- Training audit (historical evidence omitted), training record (historical evidence omitted), smoke checks (historical evidence omitted), and selection marker (historical evidence omitted)
- Operational recovery (historical evidence omitted), packaging correction (historical evidence omitted), and executed source freeze (historical evidence omitted)
- Calibration: released (historical evidence omitted) and selected baseline (historical evidence omitted)
- Held-out teacher audit (historical evidence omitted), final artifact freeze (historical evidence omitted), and capacity accounting (historical evidence omitted)

Checkpoints and GPU traces remain in ignored `.artifacts/decision-model-next-large/download/.artifacts/decision-model-next/run/`; native final traces remain in ignored `.artifacts/decision-model-next-large/final-harbor/`. No historical experiment, application default, billing-service policy or application dependency lock was changed. No commit, push or deployment was performed.

[Public setup and historical-input prerequisites](../experiments/decision_model/README.md).
