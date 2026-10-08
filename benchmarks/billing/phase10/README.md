# Phase 10 deterministic billing corpus

This directory contains 40 reviewed family specs and their 200 generated Scenario snapshots. The original 25 tasks in `benchmarks/billing/scenarios` and `benchmarks/billing/tasks` remain independent anchors. No model is trained here.

## Author and inspect

1. Edit one YAML file in `families/`. Every family names an existing synthetic prototype, five instruction variants, controlled state dimensions, a target-status constraint, and one known-wrong strategy.
2. Run `make benchmark-billing-phase10` to regenerate the whole snapshot atomically. Inspect that family's five Scenario YAMLs under `generated/scenarios/` and `generated/corpus-stats.json`.
3. Run `make benchmark-billing-phase10-check`. It regenerates in a temporary directory and checks every byte of the snapshot and manifests. Ordinary CI runs this command without Docker or AWS.
4. Run `make benchmark-billing-phase10-render-smoke` to render and validate all 200 tasks in a temporary directory. CI runs this CPU-only gate after the snapshot check.
5. Run `make benchmark-billing-phase10-render` to create 200 Harbor tasks on demand under `.rl-smoke/phase10/rendered/tasks/` and derive a hash-addressed shared-verifier execution suite.
6. On a Docker-capable host, run `make benchmark-billing-phase10-acceptance`. It requires 200 Oracle rewards of 1 and 40 representative known-wrong rewards of 0 using the existing Harbor verifier. Exact corpus and execution-suite hashes key both caches. Passing acceptance writes `.rl-smoke/phase10/acceptance/<corpus-sha>/summary.json` and `reports/phase-10-billing-corpus.md`.

`config.yaml` fixes the generator version and seed. `split.yaml` assigns complete families to train, dev, and holdout with prototype and structural behavior disjointness. The split is external to Scenario YAML. The corpus SHA covers sorted task IDs and semantic Scenario hashes; each Scenario records its prototype SHA. Phase 11 should use `load_phase10_corpus(split=...)` and preserve both corpus and split manifest SHAs as part of every experiment identity.

After Phase 11 begins, do not tune against holdout. If a task is objectively invalid, edit the family, regenerate, and treat the new corpus and split hashes as a new dataset revision.
