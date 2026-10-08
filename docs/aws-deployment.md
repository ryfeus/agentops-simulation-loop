# Optional AWS deployment

Use a dedicated sandbox account with synthetic data, budget alerts, and an operator authorized to manage IAM, ECR, AgentCore, DSQL, CloudWatch/X-Ray, and optional EC2 resources. Deployment creates billable resources. Review [cost/security](cost-and-security.md) and [operator safety](aws-safety.md) first. This implementation supports `us-west-2` only.

Install AWS CLI v2, Terraform meeting `infra/terraform/versions.tf`, Docker/buildx, jq, Python and uv. Configure your own AWS SSO profile; neither this repository nor CI supplies credentials.

```bash
uv sync --locked --all-groups --all-extras
export AWS_PROFILE=my-sandbox
export EXPECTED_AWS_ACCOUNT_ID=123456789012 # fictional example: replace it
aws sso login --profile "$AWS_PROFILE"
make aws-plan
make aws-foundation
make aws-dsql-bootstrap
make aws-agentcore-config MODEL_PROVIDER=bedrock MODEL_ID=YOUR_BEDROCK_MODEL_ID
make aws-agentcore-image
make aws-agentcore-deploy
make aws-agentcore-smoke
```

Read each Terraform plan before continuing. The wrapper verifies STS identity and exports the intended account as `TF_VAR_expected_aws_account_id`; direct Terraform usage must set that variable explicitly. The AWS provider rejects mismatches. Generated runtime configuration and images are revision-bound; commit source changes before packaging. Bedrock model availability and permissions must be configured in your account.

## Synthetic trace and evaluation loop

Ordinary deployments omit prompt/tool content and world snapshots. For the synthetic Taskify loop, explicitly opt in before deploying the runtime:

```bash
export TF_VAR_synthetic_demo_trace_content_capture=true
make aws-agentcore-deploy
make aws-observability-status
export AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE=yes
make aws-observability-bootstrap TRANSACTION_SEARCH_INDEXING_PERCENTAGE=100
make aws-observability-check TRANSACTION_SEARCH_INDEXING_PERCENTAGE=100
make aws-observability-smoke
make aws-evaluator-build
make aws-evaluation-plan
make aws-evaluation-deploy
make aws-evaluation-status
make aws-evaluation-trigger
make aws-evaluation-find-failure SESSION_ID=YOUR_SESSION TRACE_ID=YOUR_TRACE
make aws-taskify-failure
```

Transaction Search changes account/region-wide settings that affect other applications and incur indexing cost. `100` is the synthetic demo choice, not an onboarding default. Keep the generated `.agentcore/observability/` receipt so you can explicitly restore previous settings. Collect fresh failure traces after enabling capture. Billing snapshots are required; incomplete historical traces cannot be reconstructed by guessing state.

Interactive failure capture uses `aws-evaluation-find-failure` with explicit `SESSION_ID` and `TRACE_ID`; inspect the Makefile for the exact target arguments. Optional EC2 Harbor/GPU tracks require their additional network, key, instance-type and quota preflight; see [RL training](rl-training.md). Do not broaden controller CIDRs or IAM permissions to bypass failures.

## Reset and cleanup

`make aws-dsql-reset` and `make aws-reset` replace demo database state. Use them only in the isolated sandbox. Inspect active Harbor and GPU workers and their project/run tags before cleanup; terminate only owned resources through the provided controllers.

```bash
make aws-observability-restore OBSERVABILITY_RECEIPT=.agentcore/observability/RECEIPT.json
make aws-down
# If the optional GPU foundation was provisioned:
make aws-rl-smoke-down
```

Restore API-managed observability settings separately; Terraform destroy does not restore them. Retain receipts until cleanup is verified. Inspect Terraform state/output and controller status to confirm no owned EC2 instances or optional foundation resources remain. Local export validation never runs these commands.
