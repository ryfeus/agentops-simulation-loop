# Billing decision-model research

This optional research track compares released and continued Strands Decider policies with rule and Bedrock baselines in synthetic billing tasks. It does not enable a learned policy in the normal billing agent.

## Credential-free starting point

Run from the repository root:

```bash
python3 -m experiments.decision_model.demo
uv sync --locked --all-groups --all-extras
uv run pytest tests/unit/test_decision_*.py
```

The terminal demo reads retained aggregate replay metrics and synthetic examples. It downloads no models and calls no services. The root tests exercise synthetic contracts; optional tensor wiring tests require the separate training environment.

## Optional model environment

```bash
uv sync --project experiments/decision_model --locked
uv run --project experiments/decision_model python -m experiments.decision_model.prepare
```

The isolated project requires Python 3.12. Preparation downloads the pinned public source and model artifacts listed in `sources.json`; it does not train or call AWS. On a separately provisioned compatible Linux GPU worker, add `--extra gpu` during sync. Training remains outside default CI. Model artifacts retain their upstream license terms and are not distributed here.

## Historical training inputs

Historical demonstrations, source checkpoints, runtime snapshots, calibration files, cohort freezes, and execution archives are omitted from this public export. An exact historical replay requires those inputs from an authorized local source. Public metrics are results-only exports, not substitutes for integrity manifests. Do not remove hash checks or regenerate evidence merely to satisfy a historical freeze.

For an operator-supplied directory containing verified Phase 12 training demonstrations and metadata:

```bash
uv run --project experiments/decision_model python -m experiments.decision_model.data \
  --source /path/to/verified-evidence --output .artifacts/decision-model/dataset
```

Use a fresh output directory. Data extraction validates actual teacher outcomes and preserves prototype-disjoint split allocation. New authored-cohort generators and evaluation commands expose their inputs through `--help`; evaluations requiring retained historical sources remain unavailable without those sources. Reproducing a new run does not reproduce the omitted historical evidence.

## Opt-in paid execution

Bedrock inference and SSM/S3 transport require an operator-owned AWS account, credentials, and separately provisioned resources. No cloud deployment or inference is part of installation, tests, or the offline demo.

```bash
export AWS_PROFILE=my-sandbox
export EXPECTED_AWS_ACCOUNT_ID=123456789012  # fictional; replace with your account
aws sso login --profile "$AWS_PROFILE"
scripts/with_aws_credentials.sh uv run --project experiments/decision_model \
  python -m experiments.decision_model.llm --help
```

Both direct experiment inference and cloud transport validate the selected account before creating service clients. The supported region remains `us-west-2`. Cloud transport uses the existing Terraform worker outputs via `cloud init --outputs PATH`; its generated worker/bucket/command records remain under ignored `.artifacts/`. Never commit those records, bearer tokens, traces, model weights, or adapters.

Native Harbor execution additionally requires Docker, rendered tasks, a package wheel built from a clean public revision, and a current AgentConfig. Historical worker paths are not installation prerequisites. Local benchmark and demo commands are documented in the main repository guide.

## Results and interpretation

See [the research index](../../docs/research-results.md) for the initial adaptation and subsequent history, authorization, fallback, and generalization studies. Deferral, native verifier passes, strict task completion, rejected proposals, and executed writes are distinct measurements. Passing a synthetic guard test does not establish production safety or generalization.
