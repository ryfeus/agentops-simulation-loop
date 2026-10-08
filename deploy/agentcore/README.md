# AgentCore container

The Dockerfile builds the provenance-bound runtime image from the project wheel and the locked `requirements.txt`. `make agentcore-image-check` validates an ARM64 build locally without AWS; `make aws-agentcore-image` is a guarded, billable registry publication path.

See [AWS deployment](../../docs/aws-deployment.md) and [operator safety](../../docs/aws-safety.md) before using live commands. Runtime content capture and billing world snapshots are disabled unless the synthetic capture flag is explicitly enabled. The deployed candidate default is retained independently of local scripted chat.
