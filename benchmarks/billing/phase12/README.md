# Phase 12 evaluation corpus

`prototypes.json` contains 36 authored compositions: 18 development prototypes and
18 final-test prototypes. Each has five explicit instruction variants. The nine
archetypes each contribute two prototypes and ten tasks to each split.

The compositions require observable additional behavior, rather than merely new
invoice identifiers: development tasks combine the primary request with either
an open/paid companion audit or a completed-refund history audit. Final-test
tasks combine it with an existing-escalation audit or a mixed paid/refunded/
escalated history audit. Every named companion must be inspected and preserved.
These are controlled synthetic composition tests, not a sample of production
traffic. They deliberately test longer interactions, up to seven tool calls.

`rl.phase12.corpus` deterministically materializes these into existing Scenario
contracts, retaining the original Phase 10 160-task training split byte-for-byte
at the semantic level. `manifest.json` freezes all scenario, prototype,
normalized-instruction, and behavior-signature hashes. Generation does not load
models. `make rl-phase12-render-smoke` renders and validates all 340 tasks in a
temporary directory without executing Harbor.

The generator rejects cross-split family/prototype, structural behavior, and
normalized-instruction overlap. A source change requires an intentional new
experiment identity; do not regenerate the manifest to make an active run pass.

CPU tests exercise all fresh tasks through the billing service and shared
verifier, including correct, no-op, and wrong-write trajectories. Full Harbor
Docker calibration remains an explicit execution prerequisite. Its proof
contains scores and hashes, not test conversations. Training payloads contain
only train/dev task directories and never include this prototypes file.
