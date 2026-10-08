# Evaluation and evidence

The online deterministic dispute-policy evaluator reads correlated tool spans to detect a policy failure. It does not inspect an isolated database or prove that a candidate fixes the failure. Harbor's offline verifier checks final synthetic billing state, expected operations, targets, invariants, and supported trajectory requirements after executing a trial.

Taskify accepts the repository's billing trace contract: exact session/trace correlation, originating AgentConfig fingerprint, invocation instruction, ordered successful billing tool spans, and a validated pre-invocation billing world snapshot. It uses that snapshot to build the initial scenario world. It rejects missing snapshots, bad hashes, unsupported world kinds, or ambiguous correlations. Enable the synthetic capture flag and collect new evidence instead of filling gaps manually.

The [contract definitions](contracts.md) describe the normalized scenario, billing tools, candidate identity, and repository boundary. Stable task/corpus digests bind fixtures to their inputs; execution packages bind a clean Git revision and wheel hash. Replay verifies its originating candidate and evidence. Changes to source require new execution packaging, not replacing recorded provenance hashes.

```bash
make taskify-fixture
make taskify-fixture-harbor
make benchmark-billing-generated-check
make benchmark-billing-phase10-check
make benchmark-billing-phase10-render-smoke
make rl-phase12-config-check rl-phase12-render-smoke
```

These fixture and corpus commands require no cloud credentials or GPU. Harbor execution requires Docker. The registered learning gates in [research results](research-results.md) remain independent of whether a training run executed successfully. No report triggers automatic promotion or deployment.
