# Open-source sync validation

This update ports reusable decision-model research, Harbor/runtime and trajectory improvements, Bedrock benchmark aggregation, local demo utilities, and their tests onto the existing public baseline. The original MIT attribution, public onboarding, credential guards, and CI release controls are preserved.

## Included content

- Optional decision-model source and pinned isolated dependency definitions; root CI does not install the training environment or download model weights.
- Model configuration, prompt, policy, trajectory, benchmark, and short-lived demo credential support.
- Offline contract tests, synthetic temporary prior corpora, and report-processing utilities required by those tests.
- Curated historical research summaries, aggregate results-only metrics, and three synthetic offline demo examples. Private evidence is not independently reproduced by these exports.

## Exclusions and confidentiality checks

Private Git history, source revision mapping, owner account/resource identifiers, personal filesystem paths, private endpoints, credentials, signed URLs, raw trajectories/logs, operational manifests and cleanup receipts, model checkpoints/adapters, and planning documents are excluded. Only reviewed tracked files were selected. The source-to-public mapping and confidential denylist are local-only.

All experiment cloud transport and paid inference require explicit expected-account verification. Inference uses clients from that verified session, including the control-plane client; ambient credentials cannot silently override that choice. Generated cloud records remain ignored. Test accounts are fictional.

The release scanner additionally detects signed cloud URLs and private DSQL endpoints. Existing scanner exceptions remain unchanged. Gitleaks uses the public CI version and configuration; its synthetic-secret test confirms the narrow tokenizer exception still detects a secret on another line in the same file.

## Verification

- Pinned root dependency sync and isolated experiment lockfile check passed.
- `make check` passed: Ruff, frontend lint/typecheck/build, 1,592 tests passed (5 skipped, 1 deselected), scenario validation, Harbor seed/runtime gates, and AgentCore dependency checks.
- Targeted portability checks: 45 passed using generated synthetic prior corpora.
- Focused safety/model/report-scanner checks: 89 passed; credential-file cleanup checks: 4 passed.
- Isolated tensor/optimizer and checkpoint-format contract checks: 13 passed without loading model checkpoints or running a training campaign.
- Follow-up provenance compatibility tests: 19 passed. Legacy identity-only evidence remains supported; new execution settings are checked strictly, with mismatched, partial, and unknown metadata rejected.
- Scripted Docker Harbor controls passed: Oracle reward 1, known-bad reward 0, known-good reward 1.
- Offline Docker Taskify reproduction reached `VALIDATED`: Oracle 1, known-good 1, originating calibration 0. The trials ran from a clean public-only checkout; the final follow-up records validation results and curates report text.
- The offline terminal demo ran with AWS credentials and account/profile configuration removed.
- Deterministic base billing, Phase 10, and Phase 12 corpus checks and render gates passed.
- Both Terraform configurations validated with backend initialization disabled; formatting checks passed. No plan, apply, or destroy was run.
- The staged tree passed the public release scanner with a confidential local denylist. Gitleaks found no secrets in the reviewed export or the complete reviewed public branch history.

One exact historical-runtime integration test requires omitted private snapshots and is skipped when those files are absent. Tensor-specific tests are optional in the root environment; the separate tensor-wiring run passed. Missing historical inputs never bypass the production evaluator's integrity checks.

No AWS resources, paid inference, model downloads, GPU training, or external publication are part of this sync. The intended deliverable is a local review branch.
