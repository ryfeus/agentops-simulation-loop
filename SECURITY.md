# Security policy

This is a synthetic-data reference application, not a production service. Local chat is unauthenticated and intended for localhost only. Public hosting and real customer/personal/confidential datasets are outside the supported threat model.

Report vulnerabilities through this repository's GitHub **Security → Report a vulnerability** route when private vulnerability reporting is enabled. If that route is unavailable, use GitHub's private security advisory workflow with the maintainer; do not post exploit details, credentials, or private traces in a public issue. Private reporting must be enabled by the repository owner before release.

AWS deployment requires an explicit expected-account identity check and operator credentials. These controls do not replace least-privilege IAM, bounded network access, or resource ownership validation. Account-wide observability and sensitive trace capture require separate opt-ins. Temporary Bedrock role credential files are restrictive and cleaned up, but operators must also protect their host and runtime evidence.

Ordinary CI runs without AWS credentials or GPU resources. Release checks reject private metadata and require a complementary secret scan. Rotate any actually exposed credential through its owner; never include its value in a report. No production security or model-quality guarantee is made.
