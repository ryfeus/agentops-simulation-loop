# Optional RL research track

The Qwen3-0.6B / TRL / vLLM / Harbor experiments are an advanced GPU track. Default CI checks configurations and renders synthetic tasks without loading training models or provisioning instances. Begin with [local verification](quickstart.md), [evaluation](evaluation.md), and the [negative research results](research-results.md).

The isolated `rl/pyproject.toml` and `rl/uv.lock` define the pinned training environment. Model/tokenizer revisions and chat-template digests are frozen in experiment manifests. Use these files and the per-phase READMEs rather than upgrading the GPU stack independently. Reproduction requires a compatible CUDA Linux environment, Docker, and sufficient GPU memory; the AWS reference experiments use G6/L4 instances and require regional quota/capacity.

| Stage | Purpose |
| --- | --- |
| Phase 8A/8B | Validate GPU environment and Harbor-backed rollout plumbing |
| Phase 8C | Freeze a Qwen baseline and inference/execution contract |
| Phase 9/9B | Tiny overfit and persisted-adapter paired evaluation |
| Phase 10 | Freeze synthetic billing compositions and disjoint splits |
| Phase 11 | Dev-selected GRPO checkpoint with structural holdout |
| Phase 12 | Corrected baseline, SFT action gate, replicated GRPO final gate |

Safe offline checks:

```bash
make rl-smoke-config-check rl-harbor-config-check
make benchmark-billing-generated-check benchmark-billing-phase10-check
make rl-phase12-config-check rl-phase12-render-smoke
```

Cloud run/evidence targets are explicitly prefixed `aws-rl-` and require the [account guard](aws-safety.md). Inspect `Makefile`, `rl/phase8a/README.md`, and `rl/phase8b/README.md` for supported setup and commands. Later controllers live in `scripts/rl_harbor_generalization.py` and `scripts/rl_finetuning_comparison.py`; their CLI help exposes config checking, locked selection, training, evaluation and evidence operations. Do not run a final test before the prescribed selection lock or retrain after viewing it.

No weights, adapters, raw private evidence archives or checkpoints are distributed here. Historical evidence was internally verified, but a fresh export cannot independently authenticate omitted archives. Reproduction means generating new provenance-bound evidence with the published protocol, not claiming the original private run has been reproduced. GPU workers and storage remain billable until explicitly cleaned up; set budgets and verify teardown.
