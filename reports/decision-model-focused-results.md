> Public research summary. Private execution manifests, raw trajectories, checkpoints, operational receipts, and source revisions are omitted. Results below are historical measurements, not independently reproduced by this export.

# Focused decider fine-tuning: measured efficiency lift, clarification regression

Completed on 2026-10-17. A genuinely trained decider eliminated all 76 released-model fallback continuations and all 80 Bedrock calls on this focused comparison. All three arms passed every native billing check with zero unwanted writes. **The original strict no-quality-loss goal was not met:** the tuned model omitted required clarification in four task attempts where both alternatives generated a clarification question.

After reviewing these results, the user clarified that some quality loss is acceptable for demonstrating potential lift. Under that objective, this run provides positive evidence of efficiency improvement with a measured clarification tradeoff. This clarification was made after evaluation; it does not change the original acceptance result, scores, or retained failure traces.

The defensible demonstration is: **fine-tuning doubled correct direct decision coverage from 47.2% to 94.4%, reduced median policy latency by 47.5%, and preserved all required billing writes on 12 previously untouched tasks.** It also accepted eight incorrect no-action decisions, so this is evidence of efficiency potential, not a production-ready replacement or unchanged full task success.

| Arm | Native billing passes | Required writes | Unwanted writes | Invoice fallbacks | Bedrock calls | Clarification generated when required |
|---|---:|---:|---:|---:|---:|---:|
| Released decider | 24/24 | 56/56 | 0 | 76 | 80 | 4/4 |
| Tuned decider | 24/24 | 56/56 | 0 | 0 | 0 | 0/4 |
| LLM-only | 24/24 | 56/56 | 0 | 144 | 203 | 4/4 |

There were 12 tasks from six new authored composition families, two attempts per task, and three interleaved arms: 72 trials total. Tasks were held out from training, development, calibration and policy/model inference until checkpoint selection and the final freeze. All planned trials were retained; no selective retry, gold adjustment, post-test calibration or runtime repair occurred. This cohort is now exposed. Variants and attempts are correlated; these are focused within-domain results, not evidence of broad generalization or statistical noninferiority.

All 24 tuned/released and 24 tuned/LLM-only billing pairs tied: zero completion wins and zero losses. The predeclared native-only gate (at least 20% fewer fallbacks, no native completion losses, no unwanted writes) passed. The frozen reporter's `target_met: true` refers only to that gate. The supplementary acceptance record (historical evidence omitted) explicitly records the failure of the broader no-quality-loss claim.

Both deciders made **136/144 correct raw action predictions (94.4%)**. Their argmax accuracy did not improve on these held-out states. Released accepted 68 correct decisions directly and sent 76 to fallback, including all eight mistakes. Tuned accepted all 144 directly: 136 correct and eight wrong. Thus the gain came from confidence and direct execution, with an overconfidence cost. All eight wrong predictions were `no_action` where the independently authored label was `defer`; none proposed an unauthorized write.

The eight mistakes span two unresolved-choice tasks, each repeated twice, with two affected invoices per task. For example, in `focused-test-unresolved-choice-with-independent-work-v00`, the request requires a choice between invoices `inv-459212889` and `inv-459212890` before either is changed. Both models predict no action. Released confidence triggers fallback, whose retained response asks which invoice to refund. Tuned confidence is 1 and triggers no fallback, so it generates no question. This is four paired clarification-generation losses, not eight separate task failures.

Clarification here means text generated in a retained fallback response, not verified user delivery. The outer harness does not assemble or grade a complete final response; even ordinary status-reporting requests have an ungraded prose component. Do not describe billing-plus-clarification counts as fully successful conversations. Manual review also retains unsupported fallback prose, including claims about prior refund history that the tools do not expose and an active review state inferred from mere review existence. These did not cause unwanted writes. See manual trace review (historical evidence omitted) and its complete evidence (historical evidence omitted).

| Family | Native billing result | Fallbacks: released / tuned / LLM | Quality interpretation |
|---|---|---:|---|
| focused-test-absent-invoice-with-paid-write-pair | 4/4 in every arm | 14 / 0 / 24 | No billing loss; final prose not fully graded |
| focused-test-parallel-refunds-with-review-preservation | 4/4 in every arm | 12 / 0 / 24 | No billing loss; final prose not fully graded |
| focused-test-review-pair-with-three-preserved-states | 4/4 in every arm | 10 / 0 / 24 | No billing loss; final prose not fully graded |
| focused-test-three-active-operations-with-ledger-preservation | 4/4 in every arm | 8 / 0 / 24 | No billing loss; final prose not fully graded |
| focused-test-unresolved-choice-with-independent-work | 4/4 in every arm | 16 / 0 / 24 | Tuned omits clarification in all 4 paired attempts |
| focused-test-withdrawal-and-cross-operation-refund | 4/4 in every arm | 16 / 0 / 24 | No billing loss; final prose not fully graded |

| Arm | Policy seconds, median / p95 | Trial wall seconds, median / p95 | Bedrock input / output tokens |
|---|---:|---:|---:|
| Released | 13.68 / 17.85 | 19.23 / 23.21 | 146,209 / 9,246 |
| Tuned | 7.18 / 9.08 | 12.19 / 14.45 | 0 / 0 |
| LLM-only | 16.88 / 18.17 | 21.93 / 23.68 | 377,948 / 20,648 |

Median trial wall time fell 36.6% versus released. Policy time includes invoice inspections, decisions, actions and fallback; trial wall time additionally includes native environment and verification work. p95 uses nearest rank over 24 trials. Each decider processed 104,188 local input tokens. LLM-only is the same invoice-local harness with every decision delegated to Bedrock; it is not an unconstrained standalone agent benchmark. The complete run took 22.28 minutes, under the one-hour bound. The same 16-tool/16-LLM-start episode limits and one bounded continuation per invoice applied throughout. Token counts and timings are measured; a billed-dollar total was not obtained.

The run used one seed/configuration, 20 smoke steps and 200 main optimizer steps. The selected checkpoint is epoch 1, step 72; development accuracy rose from 31/40 to 40/40. Numeric weight comparison verifies changes in both the adapter and pointer head, and base-freeze instrumentation passed. All model arms used identical temperature 1 and threshold 0.7 under a protocol amendment made before test access. Original and corrected fitted calibrations are retained as secondary diagnostics. Training details (historical evidence omitted) explain selection, data, checkpoint checks and the calibration correction.

The likely failure mechanism is a gap in the defer examples. Of 144 training defer rows, 134 were synthetic missing observations; all development and calibration defer rows were synthetic. The ten observed training ambiguities used unresolved authorization conditions. In the failed test family, writes are explicitly denied pending the requested choice: the model appears to treat denied writes as finished work and misses the independent need to ask a question. That is an inference from the recorded states and training distribution, not a separately tested causal claim.

The next focused run should add contrastive examples after successful inspection: explicit read-only instructions → no action, versus “change neither until I choose; ask me which” → defer. Include these cases in development and calibration, reject checkpoints with false no-action completion there, and test on another untouched set. A response-assembly check should also verify that required questions and status summaries actually reach the user. No such repair or additional training was performed after this test was opened.

Independent audits verified all 72 records, all 216 native result/reward/diagnostic files, frozen source/config/checkpoint identities, write accounting, interleaving and bounds. There were no billing failures, blocked write attempts, provider errors, or incomplete tool evidence. The full decision suite passed 954 tests with three optional Torch skips; the 12 required isolated Torch checks passed. Native decision audit (historical evidence omitted), native integrity audit (historical evidence omitted), and full metrics (historical evidence omitted) retain the evidence.

Paid training used one g6e.16xlarge/L40S in a private research account, us-west-2. The conservative GPU occupancy upper bound is 24 minutes 2.46 seconds, within the four-hour cap; smoke took 58.89 seconds and the main loop/candidate evaluations 321.21 seconds. The GPU instance, volume, transfer bucket and all ten Terraform resources were removed and independently checked. Docker before/after inventories match exactly: all four preexisting containers remain, and no trial containers remain. Operational receipts are omitted from this public summary. No default agent behavior was changed.


Runnable training, evaluation and offline-audit commands (historical evidence omitted) and the frozen evaluation command (historical evidence omitted) reproduce the configuration. Existing evaluation paths refuse overwrite; a rerun requires new paths and must label this cohort exposed. The trained checkpoint is opt-in and has not been promoted.

[Public setup and historical-input prerequisites](../experiments/decision_model/README.md).
