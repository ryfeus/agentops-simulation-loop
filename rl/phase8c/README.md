# Phase 8C — frozen Qwen billing baseline

Phase 8C is an evaluation-only, full-suite baseline. It derives temporary
shared-verifier copies of the checked-in billing tasks and evaluates
Qwen3-0.6B plus zero-init LoRA through `GRPOTrainer.evaluate()`. It never calls
`train()`, creates an optimizer, or saves a checkpoint.

Run `make rl-harbor-baseline-config-check` before AWS work, then
`make rl-harbor-baseline-suite-check` to create or reuse an Oracle cache keyed
by the execution-suite hash. Canonical evidence lives in
`.rl-smoke/phase8c/runs/<run-id>/`; a clean revision and a complete 25 × 4 run
are required for canonical acceptance. Targeted follow-up sampling always uses
the full-suite Oracle proof, but filters only the evaluation dataset:
`PHASE8C_TASKS=task-a,task-b PHASE8C_PASSES=2 make aws-rl-harbor-baseline-run`.
Each pass remains four native TRL generations. Candidate-threshold results are
advisory and require human Phase 9 selection.
