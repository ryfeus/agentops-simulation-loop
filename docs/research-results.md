# Research results and limits

These are synthetic billing experiments, not production-readiness evidence. Successful execution, higher pass rate, and passing a registered learning gate are distinct claims.

Phase 11 selected GRPO step 320 using development prototype-macro pass rate: baseline `0.375`, step 160 `0.412`, and step 320 `0.425`. Structural holdout performance remained `0/160` before and after training; the generalization signal was false. See [the detailed Phase 11 result](../reports/phase-11-grpo-generalization.md).

Phase 12's frozen baseline final prototype-macro pass rate was `9.72%`, versus `25.14%`, `24.72%`, and `22.92%` for GRPO seeds 42, 43, and 44. Mean gain was `14.54` percentage points, with paired 95% bootstrap interval `[7.59, 21.99]`. The final learning gate failed: prohibited-write rate increased from `46.25%` to `49.72%`, `48.47%`, and `49.58%`. Learning and behavioral improvement therefore were not established under the registered criteria.

SFT failed the seed-42 action gate and was not replicated. Only GRPO received the three-seed final test, so these results do not establish a definitive replicated SFT-versus-GRPO winner. See [the detailed Phase 12 result](../reports/phase-12-sft-grpo-comparison.md) and [metrics-only JSON](../reports/phase-12-research-metrics.json).

The public reports preserve metrics, methods, limitations, and negative outcomes. Operational transfer/cleanup receipts, private source revisions, local archive paths, raw trajectories and adapters are omitted. Original evidence was internally verified; it is not included or independently reproduced by this export. The metrics JSON is a separate results-only schema, not a sanitized substitute with the original evidence hashes.

## Decision-model continuation and authorization studies

The optional decision-model research adds constrained decision replay, history visibility, authorization guards, invoice-local fallback, and successive generalization evaluations. The normal agent remains unchanged by default.

- [Initial decision adaptation](../reports/decision-model-results.md): better replay accuracy did not establish a generalization or end-to-end latency benefit.
- [History visibility](../reports/decision-model-history-results.md) and [authorization](../reports/decision-model-authorization-results.md): distinguish observed authority, rejected proposals, completed tasks, and executed writes.
- [Larger continuation study](../reports/decision-model-next-large-results.md): reports training and downstream evaluation separately.
- [Latest generalization study](../reports/decision-model-generalization-v4-results.md): both arms tied on all fresh pairs; strict completion was 8/24 per arm despite 20/24 native passes. Fewer provider calls did not establish a latency or completion improvement.

Each retained `research-metrics.json` is a results-only export of aggregate measurements. Private execution evidence and original integrity manifests are omitted. [Setup and prerequisites](../experiments/decision_model/README.md) distinguish offline checks, optional downloads, and paid execution.
