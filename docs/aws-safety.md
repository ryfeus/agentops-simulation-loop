# AWS operator safety

Local scripted demos and fixture checks need no AWS account. Cloud controllers require an explicit intended account and verify STS identity before using it. They support `AWS_PROFILE` and intentionally use `us-west-2`.

```bash
export AWS_PROFILE=my-sandbox
export EXPECTED_AWS_ACCOUNT_ID=123456789012 # synthetic example; replace with your account
aws sso login --profile "$AWS_PROFILE"
```

The shell wrapper supplies `TF_VAR_expected_aws_account_id`. For direct Terraform usage, set it explicitly to the same intended account. Terraform's AWS provider rejects a different account. Never derive the expected account automatically from the currently active caller.

Full trace content and billing world snapshots are disabled by default. To run the synthetic trace-to-Taskify demo, explicitly set `TF_VAR_synthetic_demo_trace_content_capture=true` before deployment. This captures prompts, tool inputs/outputs, and the synthetic world state; use only synthetic billing data. Collect a new trace after enabling it. Old traces without snapshots cannot become complete Taskify scenarios.

## Account-wide observability

Transaction Search affects other applications in the same account and region. Neither ordinary deployment nor Terraform destroy applies or restores its API-managed settings.

```bash
make aws-observability-status # read-only; no foundation deployment
export AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE=yes
make aws-observability-bootstrap TRANSACTION_SEARCH_INDEXING_PERCENTAGE=100
# 100% indexing is an explicit synthetic-demo choice with additional cost.
make aws-observability-check TRANSACTION_SEARCH_INDEXING_PERCENTAGE=100
make aws-observability-restore OBSERVABILITY_RECEIPT=.agentcore/observability/RECEIPT.json
```

Provision foundation resources separately before bootstrap: the CloudWatch Logs resource policy must exist. Bootstrap writes an account/region-bound `0600` receipt before changing destination or indexing. Keep that receipt locally; it records original and journaled settings, not credentials. Restore requires the same account and acknowledgement, supports journaled partial failures, and refuses conflicting state. AWS offers no compare-and-swap for these settings: use one operator and avoid concurrent changes. If an API response is ambiguous, inspect status and the retained receipt before retrying. Repeated restores are safe only while the recorded settings remain unchanged.

Local `make chat` and `make chat-cli` default to `scripted-correct`. `make aws-chat` retains its Bedrock default. Set `AGENTCORE_CANDIDATE=scripted-bad` or `bedrock` explicitly to override either; local Bedrock additionally requires `BEDROCK_MODEL_ID`.
