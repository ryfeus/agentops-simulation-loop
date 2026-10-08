# Cost and security boundaries

Use synthetic billing data only. The local browser reference app has no authentication suitable for public hosting. Bind it to localhost and do not expose it through a public tunnel or load balancer. The permissive billing server intentionally permits mistakes so evaluation can detect them.

| Surface | Risk and operator control |
| --- | --- |
| AgentCore / Bedrock | Runtime and inference charges; own account/model permissions and explicit candidate selection |
| DSQL | Billable database and destructive resets; isolated synthetic dataset and review before reset |
| CloudWatch / X-Ray | Logs, traces and indexing charges; explicit account-wide acknowledgement, chosen indexing rate, restore receipt |
| Trace content | Prompts, tool I/O and world state can disclose data; capture off by default, synthetic-only opt-in |
| Harbor EC2 | Instance, storage and networking charges; bounded concurrency, tagged ownership and explicit cleanup |
| GPU training | G6/L4 capacity, instance/storage lifecycle and long execution; quota preflight, budget alerts and verified teardown |
| Credentials | Operator-owned SSO credentials and short-lived role sessions; never commit or upload them |
| Local chat | No public authentication; localhost-only use |

No dollar estimates are supplied: costs depend on region, services, model, runtime and retention. Review current provider prices before opting in. `100%` Transaction Search indexing is an explicit demonstration setting with account-wide impact, not a safe default for a shared account.

The expected-account guard protects against the wrong active account; it does not replace least-privilege IAM, network restrictions, or ownership checks. Restore refuses state that differs from its account/region-bound receipt. There is no atomic compare-and-swap API for observability settings; avoid concurrent operators.

Terraform destroy does not restore API-managed Transaction Search settings. Optional workers and GPU foundations have separate cleanup commands. Verify their state after teardown. See [deployment](aws-deployment.md), [operator safety](aws-safety.md), and [security policy](../SECURITY.md).
