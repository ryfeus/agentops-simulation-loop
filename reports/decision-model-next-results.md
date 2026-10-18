> Public research summary. Private execution manifests, raw trajectories, checkpoints, operational receipts, and source revisions are omitted. Results below are historical measurements, not independently reproduced by this export.

# Authorization repair and second fine-tuning attempt

> Follow-up: the larger-worker continuation (historical evidence omitted) subsequently allocated an L40S and executed training. The capacity-blocked account below is preserved as the earlier attempt.

The authorization over-denial repair passes the known 24-task native Harbor regression: **24/24 completed and verified, all 30 required writes succeeded, zero unwanted writes, zero omitted operations, and zero LLM calls**. The next training pipeline and fresh family partitions are implemented, but **no new model was trained**: AWS rejected all six attempted single-L4 configurations with `Server.InsufficientInstanceCapacity`. There is no new model improvement measurement.

## Authorization result

| Known-cohort metric | Previous guarded, history visible | Corrected authorization, history visible |
|---|---:|---:|
| Completed and native verifier passed | 15/24 | 24/24 |
| Successful required writes | 18/30 | 30/30 |
| Missing required operations | 12 | 0 |
| Successful unwanted writes | 0 | 0 |
| LLM calls | 6 | 0 |

The corrected parser scopes negation to the command it governs, separates coordinated invoice commands, and distinguishes direct prohibitions from phrases such as “do not postpone.” Conditions requiring unavailable evidence still defer. Explicit prohibitions remain enforced at the execution boundary. A valid read-only decision on an open invoice can now finish that invoice and continue to later targets instead of unnecessarily entering fallback.

The decider input includes actual target-specific escalation history and authorization features derived only from the request and target identifiers. The execution guard independently recomputes permission; model-supplied features cannot authorize writes.

The comparison above is a targeted regression against the previous authorization run (historical evidence omitted), not a new blind benchmark or an isolated estimate of the parser's causal effect. These 24 tasks were inspected during repair and now belong to TRAIN. The new native run used the actual Docker/Harbor tool path, existing billing service and native verifiers, a 16-tool budget, and one attempt per task. Default agent behavior and historical experiment artifacts remain unchanged.

Nine known task pairs improved and none regressed. Median policy latency increased from 1,215.3 to 1,286.1 milliseconds; fewer LLM calls did not establish a median latency gain. The run lasted 155.46 seconds, below its ten-minute cap. All 24 owned environments were deleted, and the three pre-existing containers were preserved. Source hashes and all 24 native reward files were independently checked against the recorded outcomes.

## Ready training experiment

The new corpus contains 223 tasks: 175 TRAIN, 12 DEV, 24 calibration, and 12 TEST. TRAIN uses 151 compatible official historical training tasks plus the 24 known history tasks. Nine historical cases with incompatible operation-specific authorization semantics were excluded. Official historical development and test material is not used for training.

Fresh composition roots were assigned before variants: four DEV families, four calibration families, and six TEST families. The final test has two variants per family. These are newly authored compositions, not an external or independently authored benchmark.

All 211 train/dev/calibration teacher executions passed their native verifier. They produced 549 rows: 321 TRAIN, 71 DEV, and 157 calibration. Missing-observation and completed-action diagnostics together account for 19.85%; actual observed decision states dominate. The request parser supplies features, never gold labels. A metadata-only correction changed the serialization version to v2 while preserving every other row field, including inputs and labels. Exact rendered inputs fit the 1,024-token limit: maxima 707 TRAIN, 772 DEV, and 784 calibration tokens.

The implemented GPU sequence is:

1. Collect one released-model rollout pass on 48 TRAIN tasks; independently label the actual reached states and deduplicate them against existing supervision.
2. Run 20 smoke steps, checking checkpoint round-trip, adapter/head gradient updates, frozen base weights, and small-fixture learning.
3. Reload the released checkpoint and continue its existing adapter and pointer head with seed 42, unchanged learning rates, at most three epochs or 200 steps.
4. Compare epoch zero and trained candidates on eight family-balanced executed DEV tasks. Prioritize safety, task completion and omitted operations; use replay accuracy only as a final tie-breaker. Inference failures invalidate selection.
5. Calibrate both released and selected models on the reserved partition. Open TEST only after the completed selection marker exists; evaluate replay, all 24 option permutations on a bounded subset, and a four-arm native Harbor comparison with two attempts per task.

Development and routing selection use real SQLite tools and native verifiers without LLM fallback. The planned final hybrid comparison uses the same history, authorization guard and budgets for rules, released model, tuned model and LLM. This difference between standalone selection and hybrid evaluation remains a limitation. Temperature and routing also share a small calibration partition. Neither limitation is hidden by replay accuracy.

**The GPU stages above did not execute.** No model-driven collection, GPU smoke check, optimizer update, calibration, fresh test teacher collection, test prediction, or final four-arm comparison occurred. The independent holdout audit confirmed unchanged corpus/test hashes and no selection-complete marker. The fresh test remains available for the eventual run.

[Public setup and historical-input prerequisites](../experiments/decision_model/README.md).
